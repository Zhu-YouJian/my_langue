pub mod initialize;
pub mod diagnostic;
pub mod hover;
pub mod completion;
pub mod definition;
pub mod formatting;

pub trait Handler: Send + Sync {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value;
}
