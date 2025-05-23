//! b-p-e for Byte Pair Encoding

mod algorithm;

use crate::{
    Method, utok,
    vocab::{CollectedVocab, CompressedVocab, TokenType},
};
use std::{collections::HashSet, iter::zip, ops::Deref, pin::Pin, ptr::NonNull};

pub struct Bpe {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    _vocabs: Pin<Box<[u8]>>,
    /// 按 token 顺序保存元信息
    tokens: Box<[TokenMeta]>,
    /// 按字符串的字典序排序的 token 索引，用于从字符串二分查找 token。
    /// 建立索引时直接剔除了不可能从 piece 构造的所有单字节
    sorted_pieces: Box<[utok]>,
    /// 用于索引单字节 token，因此不需要其他元信息
    bytes: Box<[utok; 256]>,
    /// 特殊词汇表
    special: Box<[utok]>,
    /// token: <unk>
    unk: utok,
}

struct TokenMeta {
    /// 指向字符串内容的指针
    ptr: NonNull<u8>,
    /// 字符串长度
    len: u32,
    /// 字符串的合并排名，从 0 开始
    rank: u32,
}

// SAFETY: TokenMeta 中的指针是指向 Bpe 内容的自引用指针，且仅用于不可变引用。
unsafe impl Send for TokenMeta {}
unsafe impl Sync for TokenMeta {}

impl Deref for TokenMeta {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as _) }
    }
}

impl Bpe {
    /// 解析 tokenizer.model 文件并构造一个 bpe 分词器。
    pub fn from_tokenizer_model(model: &[u8]) -> Self {
        // 遍历文件，标记所有词汇的位置
        let offsets = (0..)
            .scan(0usize, |offset, _| match &model[*offset..] {
                [10, total_len, 10, content @ ..] => {
                    let total_len = *total_len as usize;
                    *offset += total_len + 2;
                    Some(&content[..total_len - 2])
                }
                [..] => None,
            })
            .collect::<Vec<_>>();
        // 产生词迭代器
        let vocabs = offsets.iter().map(|slice| {
            let &&[len, ref content @ ..] = slice else {
                unreachable!()
            };
            std::str::from_utf8(&content[..len as usize]).unwrap()
        });
        // 产生评分迭代器
        let scores = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            let ptr = slice[len + 2..].as_ptr().cast::<f32>();
            unsafe { ptr.read_unaligned() }
        });
        // 构造分词器
        Self::from_collected_vocab(
            CollectedVocab::collect(
                vocabs.into_iter().map(|s| s.as_bytes()),
                std::iter::repeat(TokenType::Normal),
                0,
            ),
            scores,
        )
    }

    pub fn new<'a>(
        vocabs: impl IntoIterator<Item = &'a str>,
        scores: impl IntoIterator<Item = f32>,
        token_type: impl IntoIterator<Item = TokenType>,
        unk: utok,
    ) -> Self {
        Self::from_collected_vocab(
            CollectedVocab::collect(vocabs.into_iter().map(|s| s.as_bytes()), token_type, unk),
            scores,
        )
    }

    fn from_collected_vocab(vocab: CollectedVocab, scores: impl IntoIterator<Item = f32>) -> Self {
        let CollectedVocab {
            vocabs,
            total_len,
            bytes,
            special,
            unk,
        } = vocab;
        let CompressedVocab { vocabs, slices } = CompressedVocab::new(&vocabs, total_len);
        // 收集合词评分
        let scores = scores.into_iter().collect::<Vec<_>>();
        assert_eq!(
            slices.len(),
            scores.len(),
            "scores size mismatch with vocab size"
        );
        // tokens 中直接引用字符串位置，绑定重新赋权并转换为整型的分词评分
        let tokens = zip(slices, rank(&scores))
            .map(|((off, len), rank)| TokenMeta {
                ptr: unsafe { NonNull::new_unchecked(vocabs[off..].as_ptr().cast_mut()) },
                len: len as _,
                rank,
            })
            .collect::<Box<_>>();
        // 对 token 按字符串的字典序排序，用于从字符串二分查找 token
        // <unk> 和 <0xyz> 不应该通过 piece 搜索到，使用 set 排除
        let bytes_set = bytes.iter().chain(&[unk]).cloned().collect::<HashSet<_>>();
        let mut sorted_pieces = (0..tokens.len() as utok)
            .filter(|i| !bytes_set.contains(i))
            .collect::<Box<_>>();
        sorted_pieces.sort_unstable_by_key(|&i| &*tokens[i as usize]);

        // println!(
        //     "Building BPE vocab, detected {} tokens, compressed to {} bytes from {total_len} bytes",
        //     tokens.len(),
        //     vocabs.len(),
        // );

        let mut ans = Self {
            _vocabs: vocabs,
            tokens,
            sorted_pieces,
            bytes,
            special,
            unk,
        };
        let inaccessible = ans.inaccessible();
        ans.special = ans.special.into_iter().chain(inaccessible).collect();
        ans
    }

    /// BPE 词表中，并非所有词都是合词规则可达的。此算法可识别“内部不可达”的 token。
    fn inaccessible(&self) -> Vec<utok> {
        self.sorted_pieces
            .iter()
            .filter_map(|&t| {
                let s = unsafe { std::str::from_utf8_unchecked(self.token(t)) };
                if self.encode(s).into_iter().nth(1).is_some() {
                    Some(t)
                } else {
                    None
                }
            })
            .collect()
    }

    /// piece -> token
    #[inline]
    fn find_piece(&self, piece: &[u8]) -> Option<utok> {
        match self
            .sorted_pieces
            .binary_search_by_key(&piece, |&i| self.token(i))
        {
            Ok(i) => Some(self.sorted_pieces[i]),
            Err(_) => match *piece {
                [b] => Some(self.bytes[b as usize]),
                [..] => None,
            },
        }
    }

    /// token id -> token meta
    #[inline(always)]
    fn token(&self, token: utok) -> &TokenMeta {
        &self.tokens[token as usize]
    }
}

impl Method for Bpe {
    #[inline]
    fn unk_token(&self) -> utok {
        self.unk
    }
    #[inline]
    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    #[inline]
    fn internal_special(&self) -> impl IntoIterator<Item = (&str, utok)> {
        self.special.iter().map(|&t| {
            let s = unsafe { std::str::from_utf8_unchecked(self.token(t)) };
            (s, t)
        })
    }
    #[inline]
    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_ {
        let mut tokenizer = self.begin_merge(text);
        while tokenizer.merge() {}
        tokenizer.into_iter()
    }
    #[inline]
    fn decode(&self, token: utok) -> &[u8] {
        self.token(token)
    }
}

/// 对一组评分排序、去重并重新赋权，转换为保持相同顺序的整型序列
fn rank(scores: &[f32]) -> impl IntoIterator<Item = u32> + '_ {
    use std::{
        cmp::Ordering,
        collections::{BTreeMap, BTreeSet},
    };

    #[derive(Debug)]
    struct FloatOrd(f32);
    impl Eq for FloatOrd {}
    impl PartialEq for FloatOrd {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.0.total_cmp(&other.0).is_eq()
        }
    }
    impl PartialOrd for FloatOrd {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for FloatOrd {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    let map = scores
        // 排序 + 去重
        .iter()
        .copied()
        .map(FloatOrd)
        .collect::<BTreeSet<_>>()
        // 重新赋权
        .into_iter()
        .rev()
        .enumerate()
        .map(|(i, f)| (f, i as u32))
        .collect::<BTreeMap<_, _>>();

    scores.iter().map(move |&f| map[&FloatOrd(f)])
}

#[cfg(test)]
mod bpe_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test() {
        if let Ok(buf) = std::fs::read("tokenizer.model") {
            let bpe = Bpe::from_tokenizer_model(&buf);
            let inaccessible = bpe.inaccessible();
            println!(
                "bpe: detected {} tokens, compressed to {} bytes",
                bpe.vocab_size(),
                bpe._vocabs.len(),
            );
            println!("inaccessible: {inaccessible:#?}")
        }
    }

    fn test_bpe() -> Bpe {
        Bpe::new(
            [
                "<unk>", //
                "a", "b", "c", "d", //
                "ab", "ac", "ad", "bd", //
                "bcd",
            ],
            [
                0., //
                1., 1., 1., 1., //
                1.1, 1.2, 1.3, 1.4, //
                10.,
            ],
            [TokenType::Normal; 10],
            0,
        )
    }

    #[test]
    fn test_bpe_new() {
        let bpe = test_bpe();
        assert_eq!(bpe.vocab_size(), 10);
    }

    #[test]
    fn test_bpe_unk_token() {
        let bpe = test_bpe();
        assert_eq!(bpe.unk_token(), 0);
    }

    #[test]
    fn test_bpe_encode() {
        let bpe = test_bpe();
        let encoded: Vec<_> = bpe.encode("abd").into_iter().collect();
        assert_eq!(encoded, [1, 8]); // Should merge "bc" and leave "a"
    }

    #[test]
    fn test_bpe_decode() {
        let bpe = test_bpe();
        assert_eq!(bpe.decode(3), b"c");
        assert_eq!(bpe.decode(6), b"ac");
        assert_eq!(bpe.decode(9), b"bcd");
        assert_eq!(bpe.decode(0), b"<unk>");
    }

    #[test]
    fn test_bpe_encode_decode() {
        let bpe = test_bpe();

        let text = "abcdx";
        let encoded: Vec<_> = bpe.encode(text).into_iter().collect();
        assert_eq!(encoded, [5, 3, 4, 0]);

        let decoded: Vec<_> = encoded
            .iter()
            .flat_map(|&t| bpe.decode(t).iter().copied())
            .collect();
        assert_eq!(std::str::from_utf8(&decoded), Ok("abcd<unk>"))
    }

    #[test]
    fn test_bpe_inaccessible() {
        let bpe = test_bpe();
        let inaccessible = bpe
            .internal_special()
            .into_iter()
            .collect::<HashMap<_, _>>();
        println!("Inaccessible tokens: {:?}", inaccessible);

        // 'd' is a single character, so it should be accessible
        assert!(
            !inaccessible.contains_key("d"),
            "Token 'd' should be accessible"
        );

        // 'bcd' cannot be formed by merging other tokens, so it should be inaccessible
        assert_eq!(
            inaccessible.get("bcd"),
            Some(&9),
            "Token 'bcd' should be inaccessible"
        );

        // 'ab' can be formed by merging 'a' and 'b', so it should be accessible
        assert!(
            !inaccessible.contains_key("ab"),
            "Token 'ab' should be accessible"
        );
    }

    #[test]
    fn test_bpe_with_byte_tokens() {
        let vocabs = ["a", "b", "<0x41>", "<0x42>"];
        let scores = [1.0, 1.0, 1.0, 1.0];
        let token_type = [
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Byte,
            TokenType::Byte,
        ];
        let bpe = Bpe::new(vocabs, scores, token_type, 0);

        let encoded: Vec<_> = bpe.encode("aAB").into_iter().collect();
        assert_eq!(encoded, [0, 2, 3], "Expected 3 tokens for input 'aAB'")
    }
}
