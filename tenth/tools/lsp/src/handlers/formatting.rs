use super::Handler;

pub struct FormattingHandler;

impl Handler for FormattingHandler {
    fn handle(&self, _params: Option<&serde_json::Value>) -> serde_json::Value {
        // Simplified: return empty TextEdit list
        // A full implementation would need a formatter
        serde_json::to_value(Vec::<serde_json::Value>::new()).unwrap()
    }
}
