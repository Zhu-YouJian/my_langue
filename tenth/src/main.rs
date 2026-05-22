use tenth::error::TenthResult;
use tenth::repl;

fn main() -> TenthResult<()> {
    repl::run_repl()
}