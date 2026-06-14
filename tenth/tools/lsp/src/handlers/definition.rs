use super::Handler;

pub struct DefinitionHandler;

impl Handler for DefinitionHandler {
    fn handle(&self, _params: Option<&serde_json::Value>) -> serde_json::Value {
        // Simplified: return empty list
        // A full implementation would need a symbol table to resolve definitions
        serde_json::to_value(Vec::<serde_json::Value>::new()).unwrap()
    }
}
