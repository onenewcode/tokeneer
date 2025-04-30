use ggus::GGufMetaError;

/// 将字符串中的所有空格替换为特殊的 Unicode 字符 U+2581（下八分之一块）
pub fn llama_escape_whitespace(text: &mut String) {
    // 使用 Rust 的 replace_all 方法替换所有空格
    *text = text.replace(" ", "\u{2581}");
}
