export interface Token {
  text: string;
  cls: string;
}

const KEYWORDS = new Set([
  "const", "let", "var", "fn", "pub", "impl", "struct", "enum", "match", "if", "else", "for",
  "while", "loop", "return", "use", "mod", "crate", "self", "Self", "mut", "ref", "async", "await",
  "def", "class", "import", "from", "function", "new", "try", "catch", "throw", "yield", "with",
  "as", "in", "of", "type", "interface", "export", "default", "extends", "true", "false", "null",
  "None", "True", "False", "nil", "undefined", "SELECT", "FROM", "WHERE",
]);

const TYPES = new Set([
  "i32", "i64", "u8", "u32", "u64", "usize", "f32", "f64", "bool", "str", "String", "Vec", "Option",
  "int", "float", "double", "char", "void", "number", "string", "boolean", "any",
]);

// One pass over comments, strings, numbers and identifiers. Deliberately small:
// it only has to read well in a chat bubble, not be a full parser.
const TOKEN_RE =
  /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/|--[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_$][\w$]*)/g;

export function tokenize(code: string): Token[] {
  const out: Token[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN_RE.lastIndex = 0;

  while ((m = TOKEN_RE.exec(code)) !== null) {
    if (m.index > last) out.push({ text: code.slice(last, m.index), cls: "" });
    const [full, comment, str, num, word] = m;
    if (comment) out.push({ text: full, cls: "text-gray-500 italic" });
    else if (str) out.push({ text: full, cls: "text-emerald-300" });
    else if (num) out.push({ text: full, cls: "text-amber-300" });
    else if (word) {
      if (KEYWORDS.has(word)) out.push({ text: full, cls: "text-brand-400" });
      else if (TYPES.has(word)) out.push({ text: full, cls: "text-info" });
      else if (/^[A-Z]/.test(word)) out.push({ text: full, cls: "text-violet-300" });
      else out.push({ text: full, cls: "" });
    }
    last = m.index + full.length;
  }
  if (last < code.length) out.push({ text: code.slice(last), cls: "" });
  return out;
}
