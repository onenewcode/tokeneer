#![feature(linked_list_cursors)]
mod bpe;
mod gpt2;
mod lpe;
mod tokeneer;
mod vocab;
pub use bpe::Bpe;
pub use gpt2::Gpt2Tokenizer;
pub use lpe::Lpe;
pub use tokeneer::Tokeneer;
pub use vocab::TokenType;

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

pub trait Method {
    fn unk_token(&self) -> utok;
    fn vocab_size(&self) -> usize;
    fn internal_special(&self) -> impl IntoIterator<Item = (&str, utok)>;
    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_;
    fn decode(&self, token: utok) -> &[u8];
}
