"""
CorpusGenerator.py
==================
Procedural Synthetic Corpus Generator for Pseudocode to Flowchart Translation.

Supported Grammar & Keywords:
    - Keywords: START, LET, INPUT, OUTPUT, IF, ELSE, END IF, FOR, END FOR, END
    - Flowchart Mapping:
        * START / END  -> Stadium node: id(["START"]), id(["END"])
        * INPUT / OUTPUT -> Parallelogram node: id[/"INPUT x"/], id[/"OUTPUT y"/]
        * LET          -> Rectangle process node: id["LET x = 10"]
        * IF           -> Diamond decision node: id{"condition"} with Yes / No branches
        * FOR          -> Diamond loop node: id{"FOR i = 1 TO n"} with Yes / No loopback

Storage Format (Raw_Corpus.json):
    {
      "corpus": [
        {
          "pseudocode": "...",
          "mermaid": "..."
        }
      ]
    }
"""

import re
import json
import random
import argparse
from typing import List, Dict, Set, Optional, Tuple, Any


# ==============================================================================
# CONFIGURABLE DEFAULTS (Can be changed directly in code or via CLI)
# ==============================================================================
DEFAULT_NUM_SAMPLES = 2000         # Number of pairs to generate
DEFAULT_MAX_TOKENS = 128          # Token restriction: 128 or 256
DEFAULT_OUTPUT_FILE = "Raw_Corpus.json"
DEFAULT_MAX_DEPTH = 3             # Max nesting depth for IF/FOR constructs


# ==============================================================================
# Tokenizer (for token restriction verification)
# ==============================================================================

def tokenize_code(text: str) -> List[str]:
    """
    Tokenizes pseudocode or Mermaid markup to accurately count tokens against
    the specified token limit.
    """
    pattern = re.compile(
        r'(?:END\s+IF|END\s+FOR)|[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[+\-*/%=<>()\[\]{}]|-->|--|\[/|/\]|\(\[|\]\)'
    )
    tokens = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens.extend(pattern.findall(line))
    return tokens


# ==============================================================================
# Mermaid Graph Builder
# ==============================================================================

class MermaidGraphBuilder:
    """
    Constructs syntactically valid Mermaid 'flowchart TD' scripts
    from control flow graph (CFG) nodes and edges.
    """

    def __init__(self):
        self._counter = 0
        self.nodes: List[str] = []
        self.edges: List[str] = []

    def new_node_id(self) -> str:
        self._counter += 1
        return f"node{self._counter}"

    def add_node(self, node_id: str, label: str, shape: str = "rectangle"):
        """
        Shapes:
            - stadium: ([label])
            - parallelogram: [/label/]
            - rectangle: [label]
            - diamond: {label}
        """
        # Escape any double quotes in label
        clean_label = label.replace('"', "'")

        if shape == "stadium":
            self.nodes.append(f'    {node_id}(["{clean_label}"])')
        elif shape == "parallelogram":
            self.nodes.append(f'    {node_id}[/"{clean_label}"/]')
        elif shape == "diamond":
            self.nodes.append(f'    {node_id}{{"{clean_label}"}}')
        else:  # rectangle
            self.nodes.append(f'    {node_id}["{clean_label}"]')

    def add_edge(self, from_id: str, to_id: str, label: str = ""):
        if label:
            self.edges.append(f'    {from_id} -- {label} --> {to_id}')
        else:
            self.edges.append(f'    {from_id} --> {to_id}')

    def render(self) -> str:
        """Renders the full Mermaid flowchart script with sequentially sorted edges."""
        parts = ["flowchart TD"]
        if self.nodes:
            parts.extend(self.nodes)

        if self.edges:
            parts.append("")  # blank separator line
            # Sort edges sequentially by source node ID and target node ID for clean Seq2Seq learning
            def edge_key(e: str) -> Tuple[int, int]:
                m = re.search(r'node(\d+).*?-->\s*node(\d+)', e)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
                return (0, 0)

            sorted_edges = sorted(self.edges, key=edge_key)
            parts.extend(sorted_edges)

        return "\n".join(parts)


# ==============================================================================
# AST Node Definitions
# ==============================================================================

class ASTNode:
    """Base class for all Pseudocode AST nodes."""
    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        raise NotImplementedError

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Compiles node to CFG.
        Returns:
            (entry_node_id, list_of_exit_transitions)
            where exit transition is (source_node_id, branch_label)
        """
        raise NotImplementedError


class StartNode(ASTNode):
    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        return [f"{indent_str * indent_level}START"]

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        nid = builder.new_node_id()
        builder.add_node(nid, "START", "stadium")
        return nid, [(nid, "")]


class EndNode(ASTNode):
    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        return [f"{indent_str * indent_level}END"]

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        nid = builder.new_node_id()
        builder.add_node(nid, "END", "stadium")
        return nid, []  # Terminal node, no outgoing exits


class InputNode(ASTNode):
    def __init__(self, var: str):
        self.var = var

    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        return [f"{indent_str * indent_level}INPUT {self.var}"]

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        nid = builder.new_node_id()
        builder.add_node(nid, f"INPUT {self.var}", "parallelogram")
        return nid, [(nid, "")]


class OutputNode(ASTNode):
    def __init__(self, expr: str):
        self.expr = expr

    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        return [f"{indent_str * indent_level}OUTPUT {self.expr}"]

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        nid = builder.new_node_id()
        builder.add_node(nid, f"OUTPUT {self.expr}", "parallelogram")
        return nid, [(nid, "")]


class LetNode(ASTNode):
    def __init__(self, var: str, expr: str):
        self.var = var
        self.expr = expr

    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        return [f"{indent_str * indent_level}LET {self.var} = {self.expr}"]

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        nid = builder.new_node_id()
        builder.add_node(nid, f"LET {self.var} = {self.expr}", "rectangle")
        return nid, [(nid, "")]


class IfNode(ASTNode):
    def __init__(self, condition: str, then_body: List[ASTNode], else_body: Optional[List[ASTNode]] = None):
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body

    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        lines = [f"{indent_str * indent_level}IF {self.condition}"]
        for stmt in self.then_body:
            lines.extend(stmt.to_pseudocode(indent_level + 1, indent_str))
        if self.else_body:
            lines.append(f"{indent_str * indent_level}ELSE")
            for stmt in self.else_body:
                lines.extend(stmt.to_pseudocode(indent_level + 1, indent_str))
        lines.append(f"{indent_str * indent_level}END IF")
        return lines

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        cond_id = builder.new_node_id()
        builder.add_node(cond_id, self.condition, "diamond")

        # Compile Then-branch
        then_entry, then_exits = compile_block_cfg(self.then_body, builder)
        builder.add_edge(cond_id, then_entry, "Yes")

        if self.else_body:
            # Compile Else-branch
            else_entry, else_exits = compile_block_cfg(self.else_body, builder)
            builder.add_edge(cond_id, else_entry, "No")
            return cond_id, then_exits + else_exits
        else:
            # No branch directly exits
            return cond_id, then_exits + [(cond_id, "No")]


class ForNode(ASTNode):
    def __init__(self, var: str, start_val: str, end_val: str, body: List[ASTNode]):
        self.var = var
        self.start_val = start_val
        self.end_val = end_val
        self.body = body

    def to_pseudocode(self, indent_level: int, indent_str: str) -> List[str]:
        lines = [f"{indent_str * indent_level}FOR {self.var} = {self.start_val} TO {self.end_val}"]
        for stmt in self.body:
            lines.extend(stmt.to_pseudocode(indent_level + 1, indent_str))
        lines.append(f"{indent_str * indent_level}END FOR")
        return lines

    def compile_cfg(self, builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
        for_id = builder.new_node_id()
        builder.add_node(for_id, f"FOR {self.var} = {self.start_val} TO {self.end_val}", "diamond")

        # Compile loop body
        body_entry, body_exits = compile_block_cfg(self.body, builder)

        # Enter loop
        builder.add_edge(for_id, body_entry, "Yes")

        # Loop back from body exits
        for src, lbl in body_exits:
            builder.add_edge(src, for_id, lbl)

        # Loop exit is the 'No' branch of the FOR condition
        return for_id, [(for_id, "No")]


def compile_block_cfg(statements: List[ASTNode], builder: MermaidGraphBuilder) -> Tuple[str, List[Tuple[str, str]]]:
    """Compiles a sequential sequence of AST statements into a linked CFG."""
    if not statements:
        # Fallback node if body is empty
        nid = builder.new_node_id()
        builder.add_node(nid, "pass", "rectangle")
        return nid, [(nid, "")]

    first_entry, pending_exits = statements[0].compile_cfg(builder)

    for stmt in statements[1:]:
        next_entry, next_exits = stmt.compile_cfg(builder)
        for src, lbl in pending_exits:
            builder.add_edge(src, next_entry, lbl)
        pending_exits = next_exits

    return first_entry, pending_exits


# ==============================================================================
# Procedural Generator
# ==============================================================================

class CorpusGenerator:
    """
    Generates varied, syntactically valid pseudocode programs with their
    matching Mermaid flowchart diagrams.
    """

    LOOP_VARS = ["i", "j", "k", "idx", "p", "row", "col"]
    GENERAL_VARS = ["n", "m", "x", "y", "a", "b", "val", "count", "sum", "total", "temp", "ans", "limit"]
    INPUT_VARS = ["n", "m", "x", "y", "val", "limit", "target"]

    ARITH_OPS = ["+", "-", "*", "/", "%"]
    COMP_OPS = ["==", "!=", "<", "<=", ">", ">="]

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        indent_size: int = 4,
        seed: Optional[int] = None
    ):
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.indent_size = indent_size
        if seed is not None:
            random.seed(seed)

    def _random_expr(self, in_scope_vars: Set[str], exclude_var: Optional[str] = None, depth: int = 0) -> str:
        usable = [v for v in in_scope_vars if v != exclude_var] if in_scope_vars else []
        if not usable:
            usable = ["1", "2", "5", "10"]

        if depth > 1 or random.random() < 0.60:
            if random.random() < 0.55 and usable:
                return random.choice(usable)
            return str(random.choice([0, 1, 2, 5, 10, 100]))

        left = self._random_expr(in_scope_vars, exclude_var=None, depth=depth + 1)
        op = random.choice(self.ARITH_OPS)
        right = self._random_expr(in_scope_vars, exclude_var=None, depth=depth + 1)
        if op in ["/", "%"] and right == "0":
            right = "2"
        return f"{left} {op} {right}"

    def _random_condition(self, in_scope_vars: Set[str]) -> str:
        usable = list(in_scope_vars) if in_scope_vars else ["i"]
        left_var = random.choice(usable)
        op = random.choice(self.COMP_OPS)

        mode = random.choice(["modulo", "boundary", "var_comp", "simple"])
        if mode == "modulo":
            mod_val = random.choice([2, 3, 5, 10])
            rhs = random.choice([0, 1])
            return f"{left_var} % {mod_val} {op} {rhs}"
        elif mode == "boundary":
            rhs = random.choice([0, 1, 10, 50, 100])
            return f"{left_var} {op} {rhs}"
        elif mode == "var_comp" and len(usable) > 1:
            other = random.choice([v for v in usable if v != left_var])
            return f"{left_var} {op} {other}"
        else:
            return f"{left_var} {op} {random.choice([0, 1, 2])}"

    def _generate_stmt(
        self,
        in_scope_vars: Set[str],
        active_loops: Set[str],
        current_depth: int
    ) -> ASTNode:
        can_nest = current_depth < self.max_depth

        weights = {"LET": 25, "OUTPUT": 20}
        if can_nest:
            weights["IF"] = 30
            weights["FOR"] = 25

        options = list(weights.keys())
        probs = [weights[k] for k in options]
        chosen = random.choices(options, weights=probs, k=1)[0]

        if chosen == "LET":
            target = random.choice(self.GENERAL_VARS)
            in_scope_vars.add(target)
            expr = self._random_expr(in_scope_vars, exclude_var=target)
            return LetNode(target, expr)

        elif chosen == "OUTPUT":
            if in_scope_vars and random.random() < 0.6:
                target = random.choice(list(in_scope_vars))
            else:
                target = str(random.choice([0, 1, "ans", "count"]))
            return OutputNode(target)

        elif chosen == "IF":
            cond = self._random_condition(in_scope_vars)
            # Then body
            then_size = random.choice([1, 2])
            then_body = [
                self._generate_stmt(in_scope_vars, active_loops, current_depth + 1)
                for _ in range(then_size)
            ]
            # Else body (70% probability)
            else_body = None
            if random.random() < 0.70:
                else_size = random.choice([1, 2])
                else_body = [
                    self._generate_stmt(in_scope_vars, active_loops, current_depth + 1)
                    for _ in range(else_size)
                ]
            return IfNode(cond, then_body, else_body)

        elif chosen == "FOR":
            avail_loops = [v for v in self.LOOP_VARS if v not in active_loops]
            loop_var = avail_loops[0] if avail_loops else f"idx_{current_depth}"
            active_loops.add(loop_var)
            in_scope_vars.add(loop_var)

            start_val = random.choice(["1", "0"])

            # Outer loop variable can bound inner loop (like FOR j = 1 TO i)
            candidates = [v for v in in_scope_vars if v != loop_var]
            if candidates and random.random() < 0.65:
                end_val = random.choice(candidates)
            else:
                end_val = random.choice(["n", "10", "count", "limit"])
                in_scope_vars.add(end_val)

            body_size = random.choice([1, 2])
            body = [
                self._generate_stmt(in_scope_vars, active_loops, current_depth + 1)
                for _ in range(body_size)
            ]

            active_loops.remove(loop_var)
            return ForNode(loop_var, start_val, end_val, body)

        return OutputNode("0")

    def generate_program_pair(self) -> Tuple[str, str, int]:
        """
        Generates a paired (pseudocode, mermaid) representation adhering to max_tokens.
        Returns:
            Tuple of (pseudocode_str, mermaid_str, token_count)
        """
        for _ in range(50):  # Retry attempts
            in_scope_vars: Set[str] = set()
            active_loops: Set[str] = set()
            inner_stmts: List[ASTNode] = []

            # Decide whether to wrap with START and END (approx 60% of time)
            has_start_end = random.random() < 0.60

            # Initial input or setup
            if random.random() < 0.75:
                init_var = random.choice(self.INPUT_VARS)
                in_scope_vars.add(init_var)
                if random.random() < 0.6:
                    inner_stmts.append(InputNode(init_var))
                else:
                    inner_stmts.append(LetNode(init_var, random.choice(["0", "1", "10"])))

            # Number of top-level constructs (1 to 3)
            num_top = random.choice([1, 2, 2, 3])
            for _ in range(num_top):
                inner_stmts.append(self._generate_stmt(in_scope_vars, active_loops, 0))

            # Assemble full AST sequence for CFG
            if has_start_end:
                full_stmts = [StartNode()] + inner_stmts + [EndNode()]
            else:
                full_stmts = inner_stmts

            # Convert AST to Pseudocode text
            indent_str = " " * self.indent_size
            pseudo_lines = []
            if has_start_end:
                pseudo_lines.append("START")
                for stmt in inner_stmts:
                    pseudo_lines.extend(stmt.to_pseudocode(1, indent_str))
                pseudo_lines.append("END")
            else:
                for stmt in inner_stmts:
                    pseudo_lines.extend(stmt.to_pseudocode(0, indent_str))

            pseudocode = "\n".join(pseudo_lines)

            # Compile AST to Mermaid Flowchart
            builder = MermaidGraphBuilder()
            _, pending_exits = compile_block_cfg(full_stmts, builder)

            # If program didn't have explicit END node, add a clean terminal END node
            if pending_exits:
                end_id = builder.new_node_id()
                builder.add_node(end_id, "END", "stadium")
                for src, lbl in pending_exits:
                    builder.add_edge(src, end_id, lbl)

            mermaid = builder.render()

            # Verify token restrictions
            tokens_pseudo = tokenize_code(pseudocode)
            tokens_mermaid = tokenize_code(mermaid)

            # Verify token length matches user specification
            if len(tokens_pseudo) <= self.max_tokens and len(tokens_mermaid) <= self.max_tokens * 2:
                return pseudocode, mermaid, len(tokens_pseudo)

        # Fallback minimal valid sample
        fallback_pseudo = "START\n    INPUT n\n    OUTPUT n\nEND"
        fallback_mermaid = "flowchart TD\n    node1([\"START\"]) --> node2[/\"INPUT n\"/]\n    node2 --> node3[/\"OUTPUT n\"/]\n    node3 --> node4([\"END\"])"
        return fallback_pseudo, fallback_mermaid, len(tokenize_code(fallback_pseudo))

    def generate_corpus(self, num_samples: int = DEFAULT_NUM_SAMPLES) -> List[Dict[str, str]]:
        """
        Generates N unique pairs of pseudocode and valid Mermaid code.
        """
        corpus = []
        seen = set()
        retries = 0
        max_retries = num_samples * 15

        while len(corpus) < num_samples and retries < max_retries:
            pseudo, mermaid, token_count = self.generate_program_pair()
            code_hash = hash(pseudo)
            if code_hash not in seen:
                seen.add(code_hash)
                corpus.append({
                    "pseudocode": pseudo,
                    "mermaid": mermaid
                })
            else:
                retries += 1

        return corpus


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic (Pseudocode, Mermaid) parallel corpus.")
    parser.add_argument("-n", "--count", type=int, default=DEFAULT_NUM_SAMPLES, help=f"Number of samples (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Token constraint (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH, help=f"Max nesting depth (default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT_FILE, help=f"Output file (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    generator = CorpusGenerator(
        max_tokens=args.max_tokens,
        max_depth=args.max_depth,
        seed=args.seed
    )

    print(f"Generating {args.count} pseudocode-to-mermaid pairs...")
    print(f"Token limit: <= {args.max_tokens} tokens")

    corpus = generator.generate_corpus(args.count)

    # Save format: JSON object with "corpus" array
    output_data = {
        "corpus": corpus
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully generated {len(corpus)} entries and saved to: {args.output}\n")

    # Preview sample #1
    if corpus:
        print("=" * 60)
        print("SAMPLE ENTRY #1:")
        print("=" * 60)
        print("[PSEUDOCODE]:\n" + corpus[0]["pseudocode"])
        print("\n[MERMAID]:\n" + corpus[0]["mermaid"])
        print("=" * 60)


if __name__ == "__main__":
    main()
