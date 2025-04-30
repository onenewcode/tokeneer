use crate::gpt2::session::{LlmTokenizerBpeSession, LlmTokenizerSpmSession};

pub const NULL: u32 = u32::MAX;
pub type TokenId = u32;

pub static QWEN: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#[derive(Debug, Clone)]
pub struct TokenData {
    pub text: String,
    pub score: f32,
    pub attribute: TokenAttribute,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum TokenAttribute {
    Undefined = 0,
    Unknown = 1 << 0,
    Unused = 1 << 1,
    Normal = 1 << 2,
    Control = 1 << 3, // SPECIAL?
    UserDefined = 1 << 4,
    Byte = 1 << 5,
    Normalized = 1 << 6,
    LStrIp = 1 << 7,
    RStrIp = 1 << 8,
    SingleWord = 1 << 9,
}

impl From<i32> for TokenAttribute {
    fn from(value: i32) -> Self {
        // 这里我们简单地使用 unsafe 将 i32 转换为 TokenAttribute
        // 因为我们已经使用 #[repr(i32)] 确保了内存布局兼容
        unsafe { std::mem::transmute(value) }
    }
}
