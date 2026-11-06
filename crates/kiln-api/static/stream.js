// Adapted from ollama/ollama's app/ui/app/src/util/jsonl-parsing.ts.
// Copyright (c) Ollama. See THIRD_PARTY_NOTICES.md.

export async function* parseJsonlFromStream(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const parseLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed);
    } catch {
      throw new Error("KILN returned an invalid newline-delimited JSON event.");
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        const finalValue = parseLine(buffer);
        if (finalValue !== null) yield finalValue;
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        const event = parseLine(line);
        if (event !== null) yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function* parseJsonlFromResponse(response) {
  if (!response.body) throw new Error("KILN returned an empty streaming response.");
  yield* parseJsonlFromStream(response.body);
}
