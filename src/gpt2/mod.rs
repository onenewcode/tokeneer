mod common;
mod session;
mod unicode;
mod untils;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, LinkedList},
};

use common::{NULL, QWEN, TokenAttribute, TokenData, TokenId};
use ggus::{GGuf, GGufMetaError, GGufMetaMapExt};
use memmap2::Mmap;
use session::{LlmTokenizerBpe, LlmTokenizerBpeSession};
use unicode::unicode_byte_to_utf8;
use untils::llama_escape_whitespace;

use crate::Method;

fn load_gpt2(gguf: &GGuf) -> HashMap<(String, String), usize> {
    gguf.tokenizer_ggml_merges()
        .unwrap()
        .map(|x| {
            let piece = x.unwrap();
            let (first, second) = piece.split_once(' ').unwrap();
            (first.to_string(), second.to_string())
        })
        .enumerate()
        .map(|(i, pair)| (pair, i))
        .collect()
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VocabType {
    None = 0, // For models without vocab
    Spm = 1,  // LLaMA tokenizer based on byte-level BPE with byte fallback
    Bpe = 2,  // GPT-2 tokenizer based on byte-level BPE
    Wpm = 3,  // BERT tokenizer based on WordPiece
    Ugm = 4,  // T5 tokenizer based on Unigram
    Rwkv = 5, // RWKV tokenizer based on greedy tokenization
}

pub struct Gpt2Tokenizer {
    pub vocab_type: VocabType,
    pub bos: u32,
    pub eos: u32,
    pub eot: u32,
    pub eom: u32,
    pub unk: u32,
    pub sep: u32,
    pub pad: u32,
    pub fim_pre: u32,
    pub fim_suf: u32,
    pub fim_mid: u32,
    pub fim_pad: u32,
    pub fim_rep: u32,
    pub fim_sep: u32,
    pub linefeed: u32,
    pub mask: u32,
    pub add_space_prefix: bool,
    pub add_bos: bool,
    pub add_eos: bool,
    pub ignore_merges: bool,
    pub clean_spaces: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,
    pub token_to_id: HashMap<String, TokenId>,
    pub special_tokens: Vec<TokenId>,
    pub id_to_token: Vec<TokenData>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub session: RefCell<LlmTokenizerBpeSession>,
}
impl Gpt2Tokenizer {
    pub fn new() -> Self {
        Self {
            vocab_type: VocabType::None,
            bos: 1,
            eos: 2,
            eot: NULL,
            eom: NULL,
            unk: 0,
            sep: NULL,
            pad: NULL,
            fim_pre: NULL,
            fim_suf: NULL,
            fim_mid: NULL,
            fim_pad: NULL,
            fim_rep: NULL,
            fim_sep: NULL,
            linefeed: NULL,
            mask: NULL,
            add_space_prefix: false,
            add_bos: true,
            add_eos: false,
            ignore_merges: false,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
            token_to_id: HashMap::new(),
            special_tokens: Vec::new(),
            id_to_token: Vec::new(),
            bpe_ranks: HashMap::new(),
            session: LlmTokenizerBpeSession::new(LlmTokenizerBpe {
                // qwen
                regex_exprs: vec![QWEN.to_string()],
            })
            .into(),
        }
    }

    //  load 函数 默认都是gpt2
    pub fn load_gguf(gguf: &GGuf<'_>) -> Gpt2Tokenizer {
        // 添加多模型支持需要根据 tokenizer_ggml_mode 和tokenizer.ggml.pre对词表进行不同的初始化

        let mut config = Gpt2Tokenizer::new();

        // 设置预设字段
        config.bos = 11;
        config.eos = 11;
        config.unk = NULL;
        config.sep = NULL;
        config.pad = NULL;
        config.mask = NULL;
        // bpe 需要预填充数据，设置字段
        config.add_space_prefix = false;
        config.clean_spaces = true;
        // gpt2 默认填充规则  LLAMA_VOCAB_PRE_TYPE_GPT2
        config.vocab_type = VocabType::Bpe;
        // 检查是是否有填充字段，

        // 加载特殊字符
        {
            // SPM进行分词需要
            config.add_space_prefix = gguf
                .get_bool("tokenizer.ggml.add_space_prefix")
                .unwrap_or(false);
            // remove_extra_whitespaces
            config.remove_extra_whitespaces = gguf
                .get_bool("tokenizer.ggml.remove_extra_whitespaces")
                .unwrap_or(false);

            let matche_token = |token: Result<u32, GGufMetaError>, target: u32| -> u32 {
                if token.is_ok() {
                    token.unwrap()
                } else {
                    target
                }
            };
            config.bos = matche_token(gguf.tokenizer_ggml_bos_token_id(), config.bos);
            config.eos = matche_token(gguf.tokenizer_ggml_eos_token_id(), config.eos);
            config.eot = matche_token(gguf.get_u32("tokenizer.ggml.eot_token_id"), config.eot);
            config.eom = matche_token(gguf.get_u32("tokenizer.ggml.eom_token_id"), config.eom);
            config.unk = matche_token(gguf.get_u32("tokenizer.ggml.unknown_token_id"), config.unk);
            config.sep = matche_token(
                gguf.get_u32("tokenizer.ggml.seperator_token_id"),
                config.sep,
            );
            config.pad = matche_token(gguf.get_u32("tokenizer.ggml.padding_token_id"), config.pad);
            config.mask = matche_token(gguf.get_u32("tokenizer.ggml.mask_token_id"), config.mask);
            config.fim_pre = matche_token(
                gguf.get_u32("tokenizer.ggml.fim_pre_token_id"),
                config.fim_pre,
            );
            config.fim_suf = matche_token(
                gguf.get_u32("tokenizer.ggml.fim_suf_token_id"),
                config.fim_suf,
            );
            config.fim_mid = matche_token(
                gguf.get_u32("tokenizer.ggml.fim_mid_token_id"),
                config.fim_mid,
            );
            config.fim_pad = matche_token(
                gguf.get_u32("tokenizer.ggml.fim_pad_token_id"),
                config.fim_pad,
            );
            config.fim_rep = matche_token(
                gguf.get_u32("tokenizer.ggml.fim_rep_token_id"),
                config.fim_rep,
            );
            config.fim_sep = matche_token(
                gguf.get_u32("tokenizer.ggml.fim_sep_token_id"),
                config.fim_sep,
            );

            config.add_bos = gguf
                .get_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(config.add_bos);
            config.add_eos = gguf
                .get_bool("tokenizer.ggml.add_eos_token")
                .unwrap_or(config.add_eos);
        }

        let tokens = gguf.tokenizer_ggml_tokens().unwrap();
        let scores = gguf
            .tokenizer_ggml_scores()
            .ok()
            .map(|arr| arr.map(|r| r.unwrap()).collect::<Vec<_>>());
        let token_type = gguf
            .tokenizer_ggml_token_type()
            .ok()
            .map(|arr| arr.map(|r| r.unwrap()).collect::<Vec<_>>())
            .unwrap();
        // 此处等同于llama.cpp的合并
        let bpe_ranks = load_gpt2(&gguf);
        let mut id_to_token = Vec::with_capacity(tokens.len());

        let mut token_to_id: HashMap<String, TokenId> = HashMap::with_capacity(tokens.len());

        for (i, text) in tokens.into_iter().enumerate() {
            let text = text.unwrap().to_string();
            let score = scores.as_ref().map_or(0.0, |s| s[i]);
            let attribute = match token_type[i] {
                1 => TokenAttribute::Normal,
                2 => TokenAttribute::Unknown,
                3 => TokenAttribute::Control,
                4 => TokenAttribute::UserDefined,
                5 => TokenAttribute::Unused,
                6 => TokenAttribute::Byte,
                _ => TokenAttribute::Undefined,
            };

            id_to_token.push(TokenData {
                text: text.clone(),
                score,
                attribute,
            });

            token_to_id.insert(text, i as u32);
        }
        config.token_to_id = token_to_id.clone();
        config.id_to_token = id_to_token.clone();

        // 待完善 linefeed_id 暂时不支持SPM  构造换行符
        match config.vocab_type {
            VocabType::None | VocabType::Bpe => {
                let ids = config.tokenize("\n", false, false);
                if ids.is_empty() {
                    config.linefeed = config.pad;
                } else {
                    config.linefeed = ids[0];
                }
            }
            VocabType::Spm => {
                config.linefeed = if token_to_id.contains_key("\n") {
                    *token_to_id.get("\n").unwrap()
                } else {
                    config.pad
                };
            }
            VocabType::Wpm => todo!(),
            VocabType::Ugm => todo!(),
            VocabType::Rwkv => todo!(),
        }

        for (key, value) in &token_to_id {
            if config.eot == NULL {
                if key == "<|eot_id|>"
                    || key == "<|im_end|>"
                    || key == "<|end|>"
                    || key == "<end_of_turn>"
                    || key == "<|endoftext|>"
                    || key == "< EOT >"
                    || key == "_< EOT >"
                    || key == "<｜end▁of▁sentence｜>"
                // DeepSeek
                {
                    config.eot = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.eom == NULL {
                if key == "<|eom_id|>" {
                    config.eom = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_pre == NULL {
                if key == "<|fim_prefix|>" // Qwen
                || key == "<fim-prefix>"
                || key == "<｜fim▁begin｜>" // DeepSeek
                || key == "<PRE>"
                || key == "▁<PRE>"
                {
                    config.fim_pre = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_suf == NULL {
                if key == "<|fim_suffix|>" // Qwen
            || key == "<fim-suffix>"
            || key == "<｜fim▁hole｜>" // DeepSeek
            || key == "<SUF>"
            || key == "▁<SUF>"
                // CodeLlama
                {
                    config.fim_suf = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_mid == NULL {
                if key == "<|fim_middle|>" // Qwen
            || key == "<fim-middle>"
            || key == "<｜fim▁end｜>" // DeepSeek
            || key == "<MID>"
            || key == "▁<MID>"
                // CodeLlama
                {
                    config.fim_mid = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_mid == NULL {
                if key== "<|fim_middle|>" // Qwen
            || key== "<fim-middle>"
            || key== "<｜fim▁end｜>"  // DeepSeek
            || key== "<MID>"
            || key== "▁<MID>"
                // CodeLlama
                {
                    config.fim_mid = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_pad == NULL {
                if key == "<|fim_pad|>" // Qwen
                || key == "<fim-pad>"
                || key == "<PAD>"
                {
                    config.fim_pad = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_rep == NULL {
                if key == "<|fim_repo|>"  // Qwen
            || key == "<|repo_name|>"
            || key == "<fim-repo>"
            || key == "<REPO>"
                {
                    config.fim_rep = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
            if config.fim_sep == NULL {
                if key == "<|file_sep|>"
                // Qwen
                {
                    config.fim_sep = *value;
                    if (id_to_token[*value as usize].attribute as i32
                        & TokenAttribute::Control as i32)
                        == 0
                    {
                        id_to_token[*value as usize].attribute = TokenAttribute::Control;
                    }
                }
            }
        }
        let mut special_eog_ids = HashSet::new();
        // maintain a list of tokens that cause end-of-generation
        if config.fim_pad != NULL && !special_eog_ids.contains(&config.fim_pad) {
            special_eog_ids.insert(config.fim_pad);
        }
        if config.fim_rep != NULL && !special_eog_ids.contains(&config.fim_rep) {
            special_eog_ids.insert(config.fim_rep);
        }
        if config.fim_sep != NULL && !special_eog_ids.contains(&config.fim_sep) {
            special_eog_ids.insert(config.fim_sep);
        }

        // 第二个循环也使用引用
        for (key, value) in &token_to_id {
            if key == "<|eot_id|>"
                || key == "<|im_end|>"
                || key == "<|end|>"
                || key == "<end_of_turn>"
                || key == "<|endoftext|>"
                || key == "<|eom_id|>"
                || key == "< EOT >"
                || key == "_< EOT >"
            {
                special_eog_ids.insert(*value);
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            } else {
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                    && !special_eog_ids.contains(value)
                {
                    log::warn!("{}", key);
                }
            }
        }

        config.special_tokens = id_to_token
            .iter()
            .enumerate() // 获取索引 (TokenId) 和 TokenData
            .filter(|(_, token_data)| {
                // 检查 token 的属性是否为 Control, UserDefined 或 Unknown
                match token_data.attribute {
                    TokenAttribute::Control
                    | TokenAttribute::UserDefined
                    | TokenAttribute::Unknown => true,
                    _ => false,
                }
            })
            .map(|(index, _)| index as TokenId) // 提取符合条件的 TokenId (索引)
            .collect(); // 收集到 Vec<TokenId> 中
        config.token_to_id = token_to_id;
        config.id_to_token = id_to_token;
        config.bpe_ranks = bpe_ranks;
        config
    }
    /// 将文本字符串转换为标记 ID
    ///
    /// 如果文本在词汇表中存在，返回对应的标记 ID
    /// 否则返回 LLAMA_TOKEN_NULL
    pub fn text_to_token(&self, text: &str) -> TokenId {
        // 在 token_to_id 映射中查找文本
        if let Some(token_id) = self.token_to_id.get(text) {
            return *token_id;
        } else {
            NULL
        }
    }
    pub fn n_tokens(&self) -> u32 {
        self.id_to_token.len() as u32
    }
    /// 添加 BOS 标记
    pub fn append_bos(&self, output: &mut Vec<TokenId>) -> bool {
        if self.add_bos {
            output.push(self.bos);
            return true;
        }
        false
    }
    /// 添加 EOS 标记
    pub fn append_eos(&self, output: &mut Vec<TokenId>) -> bool {
        if self.add_eos {
            output.push(self.eos);
            return true;
        }
        false
    }
    pub fn get_token_data(&self, id: TokenId) -> &TokenData {
        &self.id_to_token[id as usize]
    }
    /// 将单个字节转换为标记 ID
    pub fn byte_to_token(&self, ch: u8) -> TokenId {
        // 十六进制字符数组
        static HEX: &[u8; 16] = b"0123456789ABCDEF";

        match self.vocab_type {
            VocabType::Spm | VocabType::Ugm => {
                // 创建格式为 "<0xXY>" 的字符串，其中 XY 是字节的十六进制表示
                let buf = format!(
                    "<0x{}{}>",
                    HEX[(ch >> 4) as usize] as char,
                    HEX[(ch & 15) as usize] as char
                );

                // 尝试在词汇表中查找该字符串
                if let Some(token) = self.token_to_id.get(&buf) {
                    return *token;
                }

                // 如果找不到，尝试回退到仅将字节作为字符串
                let buf2 = String::from_utf8_lossy(&[ch]).to_string();

                // 使用 at 方法获取标记 ID，如果不存在则会 panic
                *self.token_to_id.get(&buf2).expect("无法找到字节对应的标记")
            }

            VocabType::Wpm | VocabType::Bpe => {
                // 对于 WPM 和 BPE 类型，使用 unicode_byte_to_utf8 函数
                let utf8_str = unicode_byte_to_utf8(ch);

                // 使用 at 方法获取标记 ID，如果不存在则会 panic
                *self
                    .token_to_id
                    .get(&utf8_str)
                    .expect("无法找到字节对应的标记")
            }

            _ => {
                // 对于其他类型，终止程序
                panic!("致命错误：不支持的词汇表类型")
            }
        }
    }
    pub fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        match self
            .bpe_ranks
            .get(&(token_left.to_string(), token_right.to_string()))
        {
            Some(rank) => *rank as i32,
            None => -1,
        }
    }
    pub fn tokenize<'a>(
        &self,
        raw_text: &'a str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let mut buffer = LinkedList::new();
        let mut output = Vec::new();
        if !raw_text.is_empty() {
            buffer.push_front(
                FragmentBufferVariant::new_raw_text(raw_text.to_string(), 0, raw_text.len() as i64)
                    .unwrap(),
            );
            self.tokenizer_st_partition(&mut buffer, parse_special);
        }
        match self.vocab_type {
            VocabType::None => todo!(),
            VocabType::Spm => {
                let mut is_prev_special = true; // prefix with space if first token
                if add_special && self.add_bos {
                    output.push(self.bos);
                    is_prev_special = true;
                }
                for fragment in buffer.iter_mut() {
                    let substring = &fragment.raw_text
                        [(fragment.offset as usize)..(fragment.offset + fragment.length) as usize];
                    let mut text = String::new();
                    if fragment.variant_type == FragmentBufferVariantType::RawText {
                        if self.add_space_prefix && is_prev_special {
                            text.push(' ');
                        }
                        text.push_str(substring);

                        llama_escape_whitespace(&mut text);
                        todo!();
                        // SPM_SESSION.get_mut().unwrap()
                        //     .tokenize(&text, &mut output);
                        is_prev_special = false;
                    } else {
                        output.push(fragment.token);
                        is_prev_special = true;
                    }
                    // 检查是否有重复的 BOS 标记
                    if add_special && self.add_bos && output.len() >= 2 && output[1] == self.bos {
                        log::warn!(
                            " Added a BOS token to the prompt as specified by the model but the prompt"
                        );
                    }

                    // 添加 EOS 标记
                    if add_special && self.add_eos {
                        output.push(self.eos);
                    }
                }
            }
            VocabType::Bpe => {
                let mut session_ref = self.session.borrow_mut();
                if add_special {
                    self.append_bos(&mut output);
                }
                for fragment in buffer.iter_mut() {
                    if fragment.variant_type == FragmentBufferVariantType::RawText {
                        let substring: String = fragment
                            .raw_text
                            .chars()
                            .skip(fragment.offset as usize)
                            .take(fragment.length as usize)
                            .collect();
                        session_ref.tokenize(substring.as_str(), &mut output, &self);
                    } else {
                    }
                }

                if add_special {
                    self.append_eos(&mut output);
                }
            }
            VocabType::Wpm => todo!(),
            VocabType::Ugm => todo!(),
            VocabType::Rwkv => todo!(),
        }
        output
    }
    /// 检查文本是否有特殊标记，如果有则将其分割
    ///
    /// 例如，将 "Hello <|eot_id|> World" 分割为 "Hello" 和 "World"
    fn tokenizer_st_partition(
        &self,
        buffer: &mut LinkedList<FragmentBufferVariant>,
        parse_special: bool,
    ) {
        // 遍历每个特殊标记
        for special_id in &self.special_tokens {
            let data = self.id_to_token[*special_id as usize].clone();
            let text = &data.text;

            // 如果不解析特殊标记且当前标记是控制标记或未知标记，则跳过
            if !parse_special
                && ((data.attribute as u32)
                    & (TokenAttribute::Control as u32 | TokenAttribute::Unknown as u32))
                    != 0
            {
                continue;
            }

            // 遍历每个文本片段
            let mut cursor = buffer.cursor_front_mut();
            while let Some(fragment) = cursor.current() {
                // 如果片段是原始文本（尚未处理）
                if fragment.variant_type == FragmentBufferVariantType::RawText {
                    let FragmentBufferVariant {
                        raw_text,
                        offset,
                        length,
                        ..
                    } = &fragment.clone();
                    let mut raw_text_base_offset = *offset;
                    let mut raw_text_base_length = *length;

                    // 在文本中循环查找特殊标记
                    loop {
                        // 在当前片段中查找特殊标记的第一次出现
                        let text_slice = &raw_text[raw_text_base_offset as usize
                            ..(raw_text_base_offset + raw_text_base_length) as usize];
                        let match_pos = text_slice.find(text);

                        // 如果没有找到，停止处理该片段
                        let match_pos = match match_pos {
                            None => break,
                            Some(pos) => raw_text_base_offset as usize + pos,
                        };

                        // 如果匹配位置在基础偏移量之后，处理左侧文本
                        if match_pos > raw_text_base_offset as usize {
                            let left_reminder_offset = raw_text_base_offset as i64;
                            let mut left_reminder_length =
                                match_pos as i64 - raw_text_base_offset as i64;

                            // 如果需要去除左侧空白
                            if (data.attribute as u32 & TokenAttribute::LStrIp as u32) != 0 {
                                while left_reminder_length > 0 {
                                    let last_char = raw_text
                                        .chars()
                                        .nth(
                                            (left_reminder_offset + left_reminder_length - 1)
                                                as usize,
                                        )
                                        .unwrap();
                                    if !last_char.is_whitespace() {
                                        break;
                                    }
                                    left_reminder_length -= 1;
                                }
                            }

                            // 插入左侧文本片段
                            if left_reminder_length > 0 {
                                cursor.insert_after(
                                    FragmentBufferVariant::new_raw_text(
                                        raw_text.clone(),
                                        left_reminder_offset,
                                        left_reminder_length,
                                    )
                                    .unwrap(),
                                );
                                cursor.move_next();
                            }
                        }

                        // 插入特殊标记
                        cursor.insert_after(FragmentBufferVariant::new_token(*special_id));
                        cursor.move_next();

                        // 处理右侧文本
                        let right_start = match_pos + text.len();
                        if right_start < (raw_text_base_offset + raw_text_base_length) as usize {
                            let mut right_reminder_offset = right_start as i64;
                            let mut right_reminder_length = raw_text_base_length
                                - ((match_pos as u64 - raw_text_base_offset as u64)
                                    + text.len() as u64);

                            // 如果需要去除右侧空白
                            if (data.attribute as u32 & TokenAttribute::RStrIp as u32) != 0 {
                                while right_reminder_length > 0 {
                                    let next_char = raw_text
                                        .chars()
                                        .nth(right_reminder_offset as usize)
                                        .unwrap();
                                    if !next_char.is_whitespace() {
                                        break;
                                    }
                                    right_reminder_offset += 1;
                                    right_reminder_length -= 1;
                                }
                            }

                            // 插入右侧文本片段
                            if right_reminder_length > 0 {
                                cursor.insert_after(
                                    FragmentBufferVariant::new_raw_text(
                                        raw_text.clone(),
                                        right_reminder_offset,
                                        right_reminder_length as i64,
                                    )
                                    .unwrap(),
                                );
                                cursor.move_next();
                            }

                            // 继续处理右侧文本
                            raw_text_base_offset = right_reminder_offset as u64;
                            raw_text_base_length = right_reminder_length;
                        } else {
                            // 删除当前片段并退出循环
                            cursor.remove_current();
                            break;
                        }
                    }
                }
                cursor.move_next();
            }
        }
    }
}
impl std::fmt::Debug for Gpt2Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpt2Tokenizer")
            // 这里只添加您想要显示的字段
            .field("vocab_type", &self.vocab_type)
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .field("eot", &self.eot)
            .field("eom", &self.eom)
            .field("unk", &self.unk)
            .field("pad", &self.pad)
            .field("linefeed", &self.linefeed)
            .field("fim_pre", &self.fim_pre)
            .field("fim_suf", &self.fim_suf)
            .field("fim_mid", &self.fim_mid)
            .field("fim_pad", &self.fim_pad)
            .field("fim_rep", &self.fim_rep)
            .field("fim_sep", &self.fim_sep)
            .field("add_bos", &self.add_bos)
            .field("add_eos", &self.add_eos)
            .field("add_space_prefix", &self.add_space_prefix)
            // 不添加您不想显示的字段：token_to_id, special_tokens, id_to_token, bpe_ranks
            .finish()
    }
}
impl Method for Gpt2Tokenizer {
    /// gpt2 没有unk 这里暂时返回0
    fn unk_token(&self) -> crate::utok {
        0
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn internal_special(&self) -> impl IntoIterator<Item = (&str, crate::utok)> {
        self.special_tokens
            .iter()
            .map(|&token_id| (self.id_to_token[token_id as usize].text.as_str(), token_id))
    }

    fn encode(&self, text: &str) -> impl IntoIterator<Item = crate::utok> + '_ {
        self.tokenize(text, true, true)
            .into_iter()
            .map(|token_id| token_id)
    }

    fn decode(&self, token: crate::utok) -> &[u8] {
        self.get_token_data(token).text.as_bytes()
    }
}
#[derive(Debug, PartialEq, Clone, Copy)]
enum FragmentBufferVariantType {
    Token,
    RawText,
}

#[derive(Debug, Clone)]
struct FragmentBufferVariant {
    variant_type: FragmentBufferVariantType,
    token: u32, // 假设 llama_token 是 i32 类型
    raw_text: String,
    offset: u64,
    length: u64,
}
impl FragmentBufferVariant {
    // 创建 Token 类型的变体
    fn new_token(token: u32) -> Self {
        Self {
            variant_type: FragmentBufferVariantType::Token,
            token,
            raw_text: String::new(),
            offset: 0,
            length: 0,
        }
    }

    // 创建 RawText 类型的变体
    fn new_raw_text(text: String, offset: i64, length: i64) -> Result<Self, &'static str> {
        // 参数验证
        if offset < 0 {
            return Err("offset must be non-negative");
        }
        if length < 1 {
            return Err("length must be positive");
        }
        if (offset + length) as usize > text.len() {
            return Err("offset + length exceeds text length");
        }

        Ok(Self {
            variant_type: FragmentBufferVariantType::RawText,
            token: NULL,
            raw_text: text,
            offset: offset as u64,
            length: length as u64,
        })
    }
}
