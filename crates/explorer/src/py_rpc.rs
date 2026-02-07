use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, Serialize)]
struct Request<'a> {
    id: u64,
    method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: u64,
    pub ok: bool,
    #[serde(default)]
    pub result: Value,
    #[serde(default)]
    pub error: Option<String>,
}

pub struct PyEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl PyEngine {
    pub fn spawn(python: &str) -> Result<Self> {
        let mut child = Command::new(python)
            .args(["-m", "tools.engine_rpc"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn python engine using '{python}'"))?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("missing stdout"))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    pub fn call(&mut self, method: &str, params: Option<Value>) -> Result<Response> {
        let id = self.next_id;
        self.next_id += 1;
        let req = Request { id, method, params };
        let line = serde_json::to_string(&req)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let mut resp_line = String::new();
        self.stdout
            .read_line(&mut resp_line)
            .context("failed reading response line")?;
        if resp_line.trim().is_empty() {
            return Err(anyhow!(
                "engine returned empty response (engine likely exited)"
            ));
        }
        let resp: Response = serde_json::from_str(&resp_line)?;
        if resp.id != id {
            return Err(anyhow!(
                "engine response id mismatch: expected {id} got {}",
                resp.id
            ));
        }
        if !resp.ok {
            return Err(anyhow!(resp
                .error
                .unwrap_or_else(|| "engine error".to_string())));
        }
        Ok(resp)
    }

    pub fn shutdown(mut self) -> Result<()> {
        let _ = self.call("shutdown", None);
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }
}
