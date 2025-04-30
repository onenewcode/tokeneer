use std::collections::HashMap;

/// 将文本按照正则表达式分割成多个部分
pub fn unicode_regex_split(text: &str, regex_exprs: &[String]) -> Vec<String> {
    // Unicode 类别
    let k_ucat_enum: HashMap<&str, u32> = [
        ("\\p{N}", unicode_cpt_flags::NUMBER),
        ("\\p{L}", unicode_cpt_flags::LETTER),
        ("\\p{P}", unicode_cpt_flags::PUNCTUATION),
        ("\\p{M}", unicode_cpt_flags::ACCENT_MARK),
        ("\\p{S}", unicode_cpt_flags::SYMBOL),
    ]
    .iter()
    .cloned()
    .collect();

    let k_ucat_cpt: HashMap<u32, u8> = [
        (unicode_cpt_flags::NUMBER, 0xD1),
        (unicode_cpt_flags::LETTER, 0xD2),
        (unicode_cpt_flags::PUNCTUATION, 0xD3),
        (unicode_cpt_flags::ACCENT_MARK, 0xD4),
        (unicode_cpt_flags::SYMBOL, 0xD5),
    ]
    .iter()
    .cloned()
    .collect();

    let k_ucat_map: HashMap<u32, &str> = [
        (unicode_cpt_flags::NUMBER, "0-9"),
        (unicode_cpt_flags::LETTER, "A-Za-z"),
        (
            unicode_cpt_flags::PUNCTUATION,
            "!-#%-*,-/:-;?-@\\[-\\]_\\{\\}",
        ),
        (unicode_cpt_flags::ACCENT_MARK, ""),
        (unicode_cpt_flags::SYMBOL, "\\$+<=>^`\\|"),
    ]
    .iter()
    .cloned()
    .collect();

    let tmp = process_regex(
        &regex_exprs[0],
        text,
        &k_ucat_enum,
        &k_ucat_cpt,
        &k_ucat_map,
    );
    unicode_byte_encoding_process(&tmp)
}

/// 处理正则表达式
fn process_regex(
    regex_expr: &str,
    text: &str,
    k_ucat_enum: &HashMap<&str, u32>,
    k_ucat_cpt: &HashMap<u32, u8>,
    k_ucat_map: &HashMap<u32, &str>,
) -> Vec<String> {
    // 生成正则表达式的折叠表示
    let mut regex_expr_collapsed = String::new();

    let mut inside = false;
    let mut i = 0;
    while i < regex_expr.len() {
        let c = regex_expr.chars().nth(i).unwrap();

        if c == '[' && (i == 0 || regex_expr.chars().nth(i - 1).unwrap() != '\\') {
            regex_expr_collapsed.push('[');
            inside = true;
            i += 1;
            continue;
        }

        if inside && c == ']' && regex_expr.chars().nth(i - 1).unwrap() != '\\' {
            regex_expr_collapsed.push(']');
            inside = false;
            i += 1;
            continue;
        }

        if i + 4 < regex_expr.len()
            && regex_expr.chars().nth(i).unwrap() == '\\'
            && regex_expr.chars().nth(i + 1).unwrap() == 'p'
            && regex_expr.chars().nth(i + 2).unwrap() == '{'
            && regex_expr.chars().nth(i + 4).unwrap() == '}'
        {
            let pat = format!("\\p{{{}}}", regex_expr.chars().nth(i + 3).unwrap());
            if let Some(&cat_flag) = k_ucat_enum.get(pat.as_str()) {
                if !inside {
                    regex_expr_collapsed.push('[');
                }

                if let Some(&cat_char) = k_ucat_cpt.get(&cat_flag) {
                    regex_expr_collapsed.push(cat_char as char);
                }

                if let Some(&cat_map) = k_ucat_map.get(&cat_flag) {
                    regex_expr_collapsed.push_str(cat_map);
                }

                if !inside {
                    regex_expr_collapsed.push(']');
                }

                i += 5;
                continue;
            }
        }
        regex_expr_collapsed.push(c);
        i += 1;
    }

    // 使用折叠的文本和正则表达式
    use fancy_regex::Regex;

    match Regex::new(&regex_expr_collapsed) {
        Ok(re) => {
            // 使用正则表达式分割文本
            let mut result = Vec::new();
            let mut last_end = 0;

            for cap_result in re.captures_iter(text) {
                if let Ok(cap) = cap_result {
                    if let Some(m) = cap.get(0) {
                        // 如果匹配前有未匹配的文本，添加到结果中
                        if m.start() > last_end {
                            result.push(text[last_end..m.start()].to_string());
                        }
                        // 添加匹配的文本
                        result.push(text[m.start()..m.end()].to_string());
                        last_end = m.end();
                    }
                }
            }

            // 添加最后一部分未匹配的文本
            if last_end < text.len() {
                result.push(text[last_end..].to_string());
            }

            result
        }
        Err(_) => panic!(), // 返回空向量表示正则表达式错误
    }
}

/// 自定义正则表达式分割实现
fn unicode_regex_split_custom(text: &str, regex_expr: &str, offsets: &[usize]) -> Vec<usize> {
    if regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    {
        unicode_regex_split_custom_gpt2(text, offsets)
    } else if regex_expr
        == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        || regex_expr
            == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    {
        unicode_regex_split_custom_llama3(text, offsets)
    } else {
        Vec::new()
    }
}

/// GPT2 系统正则表达式分割实现
fn unicode_regex_split_custom_gpt2(text: &str, offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());

    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        assert!(offset_end <= cpts.len());
        start = offset_end;

        const OUT_OF_RANGE: u32 = 0xFFFFFFFF;

        let get_cpt = |pos: usize| -> u32 {
            if offset_ini <= pos && pos < offset_end {
                cpts[pos]
            } else {
                OUT_OF_RANGE
            }
        };

        let get_flags = |pos: usize| -> unicode_cpt_flags {
            if offset_ini <= pos && pos < offset_end {
                unicode_cpt_flags_from_cpt(cpts[pos])
            } else {
                unicode_cpt_flags::default()
            }
        };

        let mut prev_end = offset_ini;
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = get_cpt(pos);
            let flags = get_flags(pos);

            // 添加标记并返回长度
            let mut add_token = |end: usize| -> usize {
                assert!(prev_end <= end && end <= offset_end);
                let len = end - prev_end;
                if len > 0 {
                    bpe_offsets.push(len);
                }
                prev_end = end;
                len
            };

            // 正则表达式: 's|'t|'re|'ve|'m|'ll|'d
            if cpt == '\'' as u32 && pos + 1 < offset_end {
                let cpt_next = get_cpt(pos + 1);
                if cpt_next == 's' as u32
                    || cpt_next == 't' as u32
                    || cpt_next == 'm' as u32
                    || cpt_next == 'd' as u32
                {
                    pos += add_token(pos + 2);
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = get_cpt(pos + 2);
                    if (cpt_next == 'r' as u32 && cpt_next_next == 'e' as u32)
                        || (cpt_next == 'v' as u32 && cpt_next_next == 'e' as u32)
                        || (cpt_next == 'l' as u32 && cpt_next_next == 'l' as u32)
                    {
                        pos += add_token(pos + 3);
                        continue;
                    }
                }
            }

            let flags2 = if cpt == ' ' as u32 {
                get_flags(pos + 1)
            } else {
                flags
            };

            // 正则表达式: <space>?\p{L}+
            if flags2.is_letter {
                pos += (cpt == ' ' as u32) as usize;
                while get_flags(pos).is_letter {
                    pos += 1;
                }
                add_token(pos);
                continue;
            }

            // 正则表达式: <space>?\p{N}+
            if flags2.is_number {
                pos += (cpt == ' ' as u32) as usize;
                while get_flags(pos).is_number {
                    pos += 1;
                }
                add_token(pos);
                continue;
            }

            // 正则表达式: <space>?[^\s\p{L}\p{N}]+
            if !(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint() != 0
            {
                pos += (cpt == ' ' as u32) as usize;
                while !(get_flags(pos).is_whitespace
                    | get_flags(pos).is_letter
                    | get_flags(pos).is_number)
                    && get_flags(pos).as_uint() != 0
                {
                    pos += 1;
                }
                add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            while get_flags(pos + num_whitespaces).is_whitespace {
                num_whitespaces += 1;
            }

            // 正则表达式: \s+(?!\S)
            if num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                add_token(pos);
                continue;
            }

            // 正则表达式: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                add_token(pos);
                continue;
            }

            // 没有匹配项
            add_token(pos + 1);
            pos += 1;
        }
    }

    bpe_offsets
}

/// LLAMA3 系统正则表达式分割实现
fn unicode_regex_split_custom_llama3(text: &str, offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());

    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        assert!(offset_end <= cpts.len());
        start = offset_end;

        const OUT_OF_RANGE: u32 = 0xFFFFFFFF;

        let get_cpt = |pos: usize| -> u32 {
            if offset_ini <= pos && pos < offset_end {
                cpts[pos]
            } else {
                OUT_OF_RANGE
            }
        };

        let get_flags = |pos: usize| -> unicode_cpt_flags {
            if offset_ini <= pos && pos < offset_end {
                unicode_cpt_flags_from_cpt(cpts[pos])
            } else {
                unicode_cpt_flags::default()
            }
        };

        let mut prev_end = offset_ini;
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = get_cpt(pos);
            let flags = get_flags(pos);

            // 添加标记并返回长度
            let mut add_token = |end: usize| -> usize {
                assert!(prev_end <= end && end <= offset_end);
                let len = end - prev_end;
                if len > 0 {
                    bpe_offsets.push(len);
                }
                prev_end = end;
                len
            };

            // 正则表达式: (?i:'s|'t|'re|'ve|'m|'ll|'d) // 不区分大小写
            if cpt == '\'' as u32 && pos + 1 < offset_end {
                let cpt_next = unicode_tolower(get_cpt(pos + 1));
                if cpt_next == 's' as u32
                    || cpt_next == 't' as u32
                    || cpt_next == 'm' as u32
                    || cpt_next == 'd' as u32
                {
                    pos += add_token(pos + 2);
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = unicode_tolower(get_cpt(pos + 2));
                    if (cpt_next == 'r' as u32 && cpt_next_next == 'e' as u32)
                        || (cpt_next == 'v' as u32 && cpt_next_next == 'e' as u32)
                        || (cpt_next == 'l' as u32 && cpt_next_next == 'l' as u32)
                    {
                        pos += add_token(pos + 3);
                        continue;
                    }
                }
            }

            // 正则表达式: [^\r\n\p{L}\p{N}]?\p{L}+
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number) {
                if flags.is_letter || get_flags(pos + 1).is_letter {
                    // 一个或多个字母
                    pos += 1;
                    while get_flags(pos).is_letter {
                        pos += 1;
                    }
                    add_token(pos);
                    continue;
                }
            }

            // 正则表达式: \p{N}{1,3}
            if flags.is_number {
                let mut ini = pos;
                while get_flags(pos).is_number {
                    if pos - ini >= 3 {
                        add_token(pos);
                        ini = pos;
                    }
                    pos += 1;
                }
                add_token(pos);
                continue;
            }

            // 正则表达式: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            let flags2 = if cpt == ' ' as u32 {
                get_flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint() != 0
            {
                pos += (cpt == ' ' as u32) as usize;
                while !(get_flags(pos).is_whitespace
                    | get_flags(pos).is_letter
                    | get_flags(pos).is_number)
                    && get_flags(pos).as_uint() != 0
                {
                    pos += 1;
                }
                let mut cpt2 = get_cpt(pos);
                while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    pos += 1;
                    cpt2 = get_cpt(pos);
                }
                add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while get_flags(pos + num_whitespaces).is_whitespace {
                let cpt2 = get_cpt(pos + num_whitespaces);
                if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }

            // 正则表达式: \s*[\r\n]+
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                add_token(pos);
                continue;
            }

            // 正则表达式: \s+(?!\S)
            if num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                add_token(pos);
                continue;
            }

            // 正则表达式: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                add_token(pos);
                continue;
            }

            // 没有匹配项
            add_token(pos + 1);
            pos += 1;
        }
    }

    bpe_offsets
}

/// Unicode 代码点标志结构体
#[derive(Default, Clone, Copy)]
pub struct unicode_cpt_flags {
    pub is_whitespace: bool,
    pub is_letter: bool,
    pub is_number: bool,
    pub is_punctuation: bool,
    pub is_symbol: bool,
    pub is_accent_mark: bool,
    pub is_lowercase: bool,
    pub is_uppercase: bool,
    pub is_nfd: bool,
}

impl unicode_cpt_flags {
    pub const UNDEFINED: Self = Self {
        is_whitespace: false,
        is_letter: false,
        is_number: false,
        is_punctuation: false,
        is_symbol: false,
        is_accent_mark: false,
        is_lowercase: false,
        is_uppercase: false,
        is_nfd: false,
    };

    pub const WHITESPACE: u32 = 1 << 0;
    pub const LETTER: u32 = 1 << 1;
    pub const NUMBER: u32 = 1 << 2;
    pub const PUNCTUATION: u32 = 1 << 3;
    pub const SYMBOL: u32 = 1 << 4;
    pub const ACCENT_MARK: u32 = 1 << 5;

    pub fn as_uint(&self) -> u32 {
        let mut result = 0;
        if self.is_whitespace {
            result |= Self::WHITESPACE;
        }
        if self.is_letter {
            result |= Self::LETTER;
        }
        if self.is_number {
            result |= Self::NUMBER;
        }
        if self.is_punctuation {
            result |= Self::PUNCTUATION;
        }
        if self.is_symbol {
            result |= Self::SYMBOL;
        }
        if self.is_accent_mark {
            result |= Self::ACCENT_MARK;
        }
        result
    }

    pub fn category_flag(&self) -> u32 {
        if self.is_letter {
            return Self::LETTER;
        }
        if self.is_number {
            return Self::NUMBER;
        }
        if self.is_punctuation {
            return Self::PUNCTUATION;
        }
        if self.is_symbol {
            return Self::SYMBOL;
        }
        if self.is_accent_mark {
            return Self::ACCENT_MARK;
        }
        0
    }
}

// 以下是辅助函数的声明，这些函数在原始代码中被调用但未在片段中定义
// 在实际实现中，您需要提供这些函数的完整实现

fn unicode_cpts_from_utf8(text: &str) -> Vec<u32> {
    text.chars().map(|c| c as u32).collect()
}
/// 将 Unicode 码点转换为 UTF-8 编码的字符串
fn unicode_cpt_to_utf8(cpt: u32) -> Result<String, &'static str> {
    let mut result = String::new();

    if cpt <= 0x7f {
        // 单字节 ASCII 字符
        result.push(cpt as u8 as char);
        return Ok(result);
    }

    if cpt >= 0x80 && cpt <= 0x7ff {
        // 双字节字符
        result.push((0xc0 | ((cpt >> 6) & 0x1f)) as u8 as char);
        result.push((0x80 | (cpt & 0x3f)) as u8 as char);
        return Ok(result);
    }

    if cpt >= 0x800 && cpt <= 0xffff {
        // 三字节字符
        result.push((0xe0 | ((cpt >> 12) & 0x0f)) as u8 as char);
        result.push((0x80 | ((cpt >> 6) & 0x3f)) as u8 as char);
        result.push((0x80 | (cpt & 0x3f)) as u8 as char);
        return Ok(result);
    }

    if cpt >= 0x10000 && cpt <= 0x10ffff {
        // 四字节字符
        result.push((0xf0 | ((cpt >> 18) & 0x07)) as u8 as char);
        result.push((0x80 | ((cpt >> 12) & 0x3f)) as u8 as char);
        result.push((0x80 | ((cpt >> 6) & 0x3f)) as u8 as char);
        result.push((0x80 | (cpt & 0x3f)) as u8 as char);
        return Ok(result);
    }

    // 无效的码点
    Err("无效的码点")
}

fn unicode_cpt_flags_from_cpt(cpt: u32) -> unicode_cpt_flags {
    // 这里需要实现从代码点获取标志的逻辑
    // 在实际实现中，您可能需要查询 Unicode 数据表
    let mut flags = unicode_cpt_flags::default();

    if (cpt >= '0' as u32 && cpt <= '9' as u32) {
        flags.is_number = true;
    } else if (cpt >= 'a' as u32 && cpt <= 'z' as u32) || (cpt >= 'A' as u32 && cpt <= 'Z' as u32) {
        flags.is_letter = true;
        if cpt >= 'a' as u32 && cpt <= 'z' as u32 {
            flags.is_lowercase = true;
        } else {
            flags.is_uppercase = true;
        }
    } else if cpt == ' ' as u32 || cpt == '\t' as u32 || cpt == '\n' as u32 || cpt == '\r' as u32 {
        flags.is_whitespace = true;
    } else if cpt >= 33 && cpt <= 47
        || cpt >= 58 && cpt <= 64
        || cpt >= 91 && cpt <= 96
        || cpt >= 123 && cpt <= 126
    {
        flags.is_punctuation = true;
    }

    flags
}

fn unicode_tolower(cpt: u32) -> u32 {
    // 简单的小写转换实现
    if cpt >= 'A' as u32 && cpt <= 'Z' as u32 {
        return cpt + ('a' as u32 - 'A' as u32);
    }
    cpt
}

pub fn unicode_byte_to_utf8(byte: u8) -> String {
    let map = unicode_byte_to_utf8_map();
    map.get(&byte).unwrap().to_string()
}
/// 创建一个从字节到 UTF-8 字符串的映射
fn unicode_byte_to_utf8_map() -> HashMap<u8, char> {
    let mut map = HashMap::new();

    // 映射 ASCII 可打印字符 '!' 到 '~'
    for ch in 0x21..=0x7E {
        map.insert(ch as u8, char::from_u32(ch).unwrap());
    }

    // 映射拉丁字符 '¡' 到 '¬'
    for ch in 0xA1..=0xAC {
        map.insert(ch as u8, char::from_u32(ch).unwrap());
    }

    // 映射拉丁字符 '®' 到 'ÿ'
    for ch in 0xAE..=0xFF {
        map.insert(ch as u8, char::from_u32(ch).unwrap());
    }

    // 为剩余的字节值分配映射
    let mut n = 0;
    for ch in 0..256 {
        if !map.contains_key(&(ch as u8)) {
            map.insert(ch as u8, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    map
}
fn unicode_byte_encoding_process(bpe_words: &[String]) -> Vec<String> {
    bpe_words
        .into_iter()
        .map(|word: &String| {
            word.chars()
                .map(|ch| unicode_byte_to_utf8(ch as u8))
                .collect()
        })
        .collect()
}

/// 获取 UTF-8 字符的长度
pub fn unicode_len_utf8(byte: u8) -> usize {
    if byte & 0x80 == 0 {
        1
    } else if byte & 0xE0 == 0xC0 {
        2
    } else if byte & 0xF0 == 0xE0 {
        3
    } else if byte & 0xF8 == 0xF0 {
        4
    } else {
        1 // 无效的 UTF-8 序列，返回 1
    }
}
