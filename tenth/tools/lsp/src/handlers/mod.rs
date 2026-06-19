pub mod initialize;
pub mod diagnostic;
pub mod hover;
pub mod completion;
pub mod definition;
pub mod formatting;
pub mod document_symbol;
pub mod references;
pub mod rename;
pub mod signature_help;
pub mod folding_range;
pub mod semantic_tokens;

pub trait Handler: Send + Sync {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value;
}
