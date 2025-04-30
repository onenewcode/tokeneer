use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, VecDeque},
};

use super::{
    Gpt2Tokenizer,
    common::{NULL, TokenId},
    unicode::{unicode_len_utf8, unicode_regex_split},
};

/// 符号结构体，表示文本中的一个符号
#[derive(Clone, Debug)]
pub struct LlmSymbol {
    /// 前一个符号的索引
    pub prev: i32,
    /// 下一个符号的索引
    pub next: i32,
    /// 指向原始文本的指针
    pub text: String,
    /// 符号的长度
    pub n: usize,
}

/// BPE 标记器会话结构体

pub struct LlmTokenizerBpeSession {
    /// 标记器引用
    tokenizer: LlmTokenizerBpe,
    /// 符号列表
    symbols: Vec<LlmSymbol>,
    /// 最终符号列表
    symbols_final: Vec<LlmSymbol>,
    /// 工作队列
    work_queue: LlmBigramBpe,
}

impl LlmTokenizerBpeSession {
    /// 创建一个新的 BPE 标记器会话
    pub fn new(tokenizer: LlmTokenizerBpe) -> Self {
        Self {
            tokenizer,
            symbols: Vec::new(),
            symbols_final: Vec::new(),
            work_queue: LlmBigramBpe::new(),
        }
    }

    /// 添加标记到输出
    pub fn append(token_id: TokenId, output: &mut Vec<TokenId>) {
        output.push(token_id);
    }

    /// 标记化文本
    pub fn tokenize(&mut self, text: &str, output: &mut Vec<TokenId>, config: &Gpt2Tokenizer) {
        let mut final_prev_index = -1;
        let word_collection = unicode_regex_split(text, &self.tokenizer.regex_exprs);
        self.symbols_final.clear();

        for word in word_collection {
            self.work_queue = LlmBigramBpe::new();
            self.symbols.clear();
            // 如果词汇表忽略合并且单词已经在词汇表中
            if config.ignore_merges && config.text_to_token(&word) != NULL {
                todo!();
                self.symbols.push(LlmSymbol {
                    prev: -1,
                    next: -1,
                    text: word.to_string(),
                    n: word.len(),
                });
            }

            // 将单词分割为 UTF-8 字符
            for (i, c) in word.chars().enumerate() {
                let sym = LlmSymbol {
                    text: c.to_string(),
                    n: c.len_utf8(),
                    prev: i as i32 - 1,
                    next: if i == word.chars().count() - 1 {
                        -1
                    } else {
                        i as i32 + 1
                    },
                };
                self.symbols.push(sym);
            }

            // 添加所有可能的二元组
            for i in 1..(self.symbols.len() as i32) {
                self.add_new_bigram(i - 1, i, config);
            }
            // 构建标记
            while let Some(bigram) = self.work_queue.pop_move() {
                let left_idx = bigram.left as usize;
                let right_idx = bigram.right as usize;

                // 获取左右符号的引用
                let left_symbol = &self.symbols[left_idx];
                let right_symbol = &self.symbols[right_idx];
                let flag = format!("{}{}", &left_symbol.text, &right_symbol.text);

                // 如果其中一个符号已经被合并，跳过它
                if left_symbol.n == 0 || right_symbol.n == 0 {
                    continue;
                }

                // 检查二元组是否过时
                if flag != bigram.text {
                    continue;
                }

                // 合并右符号到左符号
                self.symbols[left_idx].n += self.symbols[right_idx].n;

                // 将右符号标记为已合并
                self.symbols[right_idx].n = 0;

                // 从链中移除右符号
                let right_next = self.symbols[right_idx].next;
                self.symbols[left_idx].next = right_next;
                self.symbols[left_idx].text = flag;
                if right_next >= 0 {
                    self.symbols[right_next as usize].prev = bigram.left;
                }
                // 寻找更多合并
                self.add_new_bigram(self.symbols[left_idx].prev, bigram.left, config);
                self.add_new_bigram(bigram.left, self.symbols[left_idx].next, config);
            }

            // 将完成的标记添加到最终列表，保持正确的顺序
            for sym in &self.symbols {
                if sym.n > 0 {
                    let mut new_sym = sym.clone();
                    new_sym.prev = final_prev_index;
                    new_sym.next = -1;

                    if final_prev_index != -1 {
                        self.symbols_final[final_prev_index as usize].next =
                            self.symbols_final.len() as i32;
                    }

                    self.symbols_final.push(new_sym);
                    final_prev_index = (self.symbols_final.len() - 1) as i32;
                }
            }
        }

        // 使用最终符号列表
        self.symbols = self.symbols_final.clone();

        // 处理所有符号
        if !self.symbols.is_empty() {
            let mut i = 0;
            while i != -1 {
                let symbol = &self.symbols[i as usize];
                if symbol.n > 0 {
                    // 创建符号的字符串
                    let str =
                        String::from_utf8_lossy(&symbol.text.as_bytes()[..symbol.n]).to_string();
                    let token = config.text_to_token(&str);

                    if token == NULL {
                        // 如果找不到标记，将每个字节作为单独的标记输出
                        for byte in str.bytes() {
                            let byte_str = String::from(byte as char);
                            let token_multibyte = config.text_to_token(&byte_str);
                            if token_multibyte != NULL {
                                output.push(token_multibyte);
                            }
                        }
                    } else {
                        // 添加找到的标记
                        output.push(token);
                    }
                }
                i = symbol.next;
            }
        }
    }

    /// 添加新的二元组
    pub fn add_new_bigram(&mut self, left: i32, right: i32, config: &Gpt2Tokenizer) {
        if left == -1 || right == -1 {
            return;
        }

        let left_token = &self.symbols[left as usize].text;
        let right_token = &self.symbols[right as usize].text;

        let rank_found = config.find_bpe_rank(left_token, right_token);

        if rank_found < 0 {
            return;
        }

        let bigram = LlmBigramBpeItem {
            left,
            right,
            text: format!("{}{}", left_token, right_token),
            size: left_token.len() + right_token.len(),
            rank: rank_found,
        };

        self.work_queue.push(bigram);
    }
}

/// BPE 二元组项结构体
#[derive(Clone, Debug)]
pub struct LlmBigramBpeItem {
    /// 左侧符号的索引
    pub left: i32,
    /// 右侧符号的索引
    pub right: i32,
    /// 二元组的文本
    pub text: String,
    /// 二元组的排名
    pub rank: i32,
    /// 二元组的大小
    pub size: usize,
}

/// BPE 二元组优先队列
#[derive(Debug)]
pub struct LlmBigramBpe {
    /// 内部队列（最小堆）
    queue: BinaryHeap<Reverse<LlmBigramBpeItem>>,
}

/// 为 LlmBigramBpeItem 实现 PartialEq
impl PartialEq for LlmBigramBpeItem {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}

/// 为 LlmBigramBpeItem 实现 Eq
impl Eq for LlmBigramBpeItem {}

/// 为 LlmBigramBpeItem 实现 PartialOrd
impl PartialOrd for LlmBigramBpeItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 为 LlmBigramBpeItem 实现 Ord，用于优先队列
impl Ord for LlmBigramBpeItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl LlmBigramBpe {
    /// 创建一个新的 BPE 二元组优先队列
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }

    /// 添加二元组到队列
    pub fn push(&mut self, item: LlmBigramBpeItem) {
        self.queue.push(Reverse(item));
    }

    /// 弹出并移动二元组
    pub fn pop_move(&mut self) -> Option<LlmBigramBpeItem> {
        self.queue.pop().map(|Reverse(item)| item)
    }

    /// 检查队列是否为空
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

///  BPE 标记器结构体
pub struct LlmTokenizerBpe {
    /// 正则表达式列表
    pub regex_exprs: Vec<String>,
}
/// 二元组结构体，用于表示两个相邻的符号
#[derive(Clone, Debug)]
pub struct LlmBigramSpm {
    /// 左侧符号的索引
    pub left: i32,
    /// 右侧符号的索引
    pub right: i32,
    /// 二元组的分数
    pub score: f32,
    /// 二元组的大小
    pub size: usize,
}

/// 为 LlmBigramSpm 实现 PartialEq
impl PartialEq for LlmBigramSpm {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}

/// 为 LlmBigramSpm 实现 Eq
impl Eq for LlmBigramSpm {}

/// 为 LlmBigramSpm 实现 PartialOrd
impl PartialOrd for LlmBigramSpm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 为 LlmBigramSpm 实现 Ord，用于优先队列
impl Ord for LlmBigramSpm {
    fn cmp(&self, other: &Self) -> Ordering {
        // 注意：这里是反向比较，因为我们需要最大堆
        // 首先比较分数，然后比较左侧索引
        match other.score.partial_cmp(&self.score) {
            Some(Ordering::Equal) => other.left.cmp(&self.left),
            Some(ord) => ord,
            None => Ordering::Equal, // 处理 NaN 情况
        }
    }
}

/// SPM 标记器会话结构体
#[derive(Debug)]
pub struct LlmTokenizerSpmSession {
    /// 符号列表
    symbols: Vec<LlmSymbol>,
    /// 工作队列
    work_queue: BinaryHeap<LlmBigramSpm>,
    /// 反向合并映射
    rev_merge: HashMap<String, (i32, i32)>,
}

impl<'a> LlmTokenizerSpmSession {
    /// 创建一个新的 SPM 标记器会话
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    /// 标记化文本
    pub fn tokenize(&mut self, text: &'a str, output: &mut Vec<u32>, config: &Gpt2Tokenizer) {
        // 将字符串分割为 UTF-8 字符
        let mut index = 0;
        let mut offs = 0;

        self.symbols.clear();

        while offs < text.len() {
            // 获取当前字符的 UTF-8 长度
            let len = unicode_len_utf8(text.as_bytes()[offs]);

            // 创建新的符号
            let sym = LlmSymbol {
                text: text.to_string(),
                n: std::cmp::min(len, text.len() - offs),
                prev: index - 1,
                next: if offs + len >= text.len() {
                    -1
                } else {
                    index + 1
                },
            };

            offs += sym.n;
            index += 1;
            self.symbols.push(sym);
        }

        // 用所有可能的 2 字符标记初始化工作队列
        for i in 1..self.symbols.len() {
            self.try_add_bigram(i as i32 - 1, i as i32, config);
        }

        // 持续替换频率最高的对，直到不能再替换
        while let Some(bigram) = self.work_queue.pop() {
            let left_idx = bigram.left as usize;
            let right_idx = bigram.right as usize;

            // 获取左右符号的可变引用
            // 注意：这里需要小心处理可变借用规则
            let left_sym_n = self.symbols[left_idx].n;
            let right_sym_n = self.symbols[right_idx].n;

            // 如果其中一个符号已经被合并，跳过它
            if left_sym_n == 0 || right_sym_n == 0 || left_sym_n + right_sym_n != bigram.size {
                continue;
            }

            // 将右符号合并到左符号中
            self.symbols[left_idx].n += right_sym_n;
            self.symbols[right_idx].n = 0;

            // 从链中移除右符号
            let right_next = self.symbols[right_idx].next;
            self.symbols[left_idx].next = right_next;

            if right_next >= 0 {
                self.symbols[right_next as usize].prev = bigram.left;
            }

            // 寻找更多替换
            self.try_add_bigram(self.symbols[left_idx].prev, bigram.left, config);
            self.try_add_bigram(bigram.left, self.symbols[left_idx].next, config);
        }

        // 处理最终的符号
        let mut i = 0;
        while i != -1 {
            let symbol = &self.symbols[i as usize];
            self.resegment(symbol, output, config);
            i = symbol.next;
        }
    }

    /// 尝试添加新的二元组
    fn try_add_bigram(&mut self, left: i32, right: i32, config: &Gpt2Tokenizer) {
        if left == -1 || right == -1 {
            return;
        }

        // 获取左右符号的文本
        let left_sym = &self.symbols[left as usize];
        let right_sym = &self.symbols[right as usize];

        // 构建完整的文本
        let left_text = &left_sym.text[..left_sym.n];
        let right_text = &right_sym.text[..right_sym.n];
        let text = format!("{}{}", left_text, right_text);

        // 查找标记
        let token = config.text_to_token(&text);

        if token == NULL {
            return;
        }

        if token as u32 >= config.n_tokens() {
            return;
        }

        // 获取标记数据
        let tok_data = config.get_token_data(token);

        // 创建新的二元组
        let bigram = LlmBigramSpm {
            left,
            right,
            score: tok_data.score,
            size: text.len(),
        };

        // 添加到工作队列
        self.work_queue.push(bigram);

        // 添加到反向合并映射
        self.rev_merge.insert(text, (left, right));
    }

    /// 重新分割符号
    fn resegment(&self, symbol: &LlmSymbol, output: &mut Vec<u32>, config: &Gpt2Tokenizer) {
        // 获取符号的文本
        let text = &symbol.text[..symbol.n];

        // 尝试将文本转换为标记
        let token = config.text_to_token(text);

        // 如果找到了标记，直接添加
        if token != NULL {
            output.push(token);
            return;
        }

        // 查找反向合并映射
        if let Some(&(left, right)) = self.rev_merge.get(text) {
            // 递归处理左右符号
            self.resegment(&self.symbols[left as usize], output, config);
            self.resegment(&self.symbols[right as usize], output, config);
            return;
        }

        // 如果没有找到映射，将每个字节作为单独的标记输出
        for i in 0..symbol.n {
            if let Some(byte) = symbol.text.as_bytes().get(i) {
                let id = config.byte_to_token(*byte);
                output.push(id);
            }
        }
    }
}
