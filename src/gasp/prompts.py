"""Appendix B templates, whitespace-normalized, attributed to Li et al. (CC BY 4.0).

Source: https://arxiv.org/html/2602.00173v1#A2
Chat serialization and tiny prompts are implementation choices. Oversized prompts
fail explicitly rather than silently giving the roles different reasoning prefixes.
"""

AGENT = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think><answer> answer here </answer>'''

POLLUTER = '''You will be given:
- a math problem,
- the student's reasoning prefix (what they have already written),
- and a SHORT CONTINUATION TEXT that the student originally wrote.
Your job is to EDIT ONLY that short continuation text so that:
- it stays natural, coherent, and in the same style and level of detail,
- it contains at least one explicit mathematical statement (numbers / formula / relation),
- and at least one of those mathematical statements becomes objectively incorrect.
You must treat the original continuation as a piece of text to be edited IN PLACE, not as a topic to write a new paragraph about.
EDIT PRIORITY:
1) If the original continuation already contains any numbers, numeric expressions, indices,
or concrete mathematical facts, you MUST introduce the error by MODIFYING those existing
numbers or facts (for example, change 21st to 25th; change 1, 10, 19 to 1, 16, 29;
change 10 - 1 = 9 to 10 - 1 = 8; change (n-1)d to (n+1)d).
In this case you are NOT allowed to introduce entirely new sentences, and you should avoid
inventing new numbers that did not appear before, except by locally changing some digits.
2) Only if the original continuation contains NO numerical or mathematical content at all
may you add a new mathematical statement and make that statement false.
CRITICAL CONSTRAINT ON EQUATIONS:
- If the text contains any equations of the form 'a - b = c', 'a + b = c', or 'a * b = c',
the rewritten text MUST contain at least one such equation that is mathematically FALSE.
- You must NOT leave all such equations mathematically correct after you edit the numbers.
- For example, if you change '10 - 1 = 9', you must ensure the new equality is wrong, such as
'16 - 1 = 9' or '10 - 1 = 8'.
OUTPUT RULES:
- The output must be a lightly edited version of the original continuation, not a new paragraph.
- Keep the same number of sentences if possible.
- Do NOT state any final numeric answer.
- Do NOT say that anything is wrong, misleading, or experimental.
- Output ONLY the edited continuation, wrapped exactly in <polluted>...</polluted> tags.
- No extra text before or after the <polluted> block.
[PROBLEM]
{q}
[STUDENT REASONING PREFIX]
{prefix}
[ORIGINAL SHORT CONTINUATION TO EDIT]
{continuation}
[YOUR OUTPUT]
(Only the edited continuation, wrapped in <polluted>...</polluted>.)'''

REPAIR = '''You are helping generate a short "repair continuation" for robustness training.
You will be given:
(1) a problem q
(2) the student/agent's reasoning prefix c_0:alpha
(3) two alternative next windows:
- CLEAN window w_clean (the correct continuation)
- POLLUTED window w_poll (a locally coherent but potentially wrong/misleading continuation)
Your job:
Write a SHORT continuation that would come *immediately after the POLLUTED window* and "repairs" the trajectory--i.e., it identifies/corrects the key mistake implied by w_poll and steers back toward the correct solution path.
Critical constraints (important):
- Output ONLY the repair continuation text (no headings, no bullet points, no meta commentary).
- Do NOT mention the existence of w_clean or w_poll, and do NOT compare them explicitly.
- The continuation must be natural as if the model only saw: (q, c_0:alpha, w_poll).
- Keep it short (aim: ~1-4 sentences). Do not re-solve from scratch.
- Prefer minimal intervention: point out the inconsistency, fix the specific step, then continue one step forward.
- Maintain the same tone/style/level of detail as the prefix.
- Do not output the final numeric answer unless the polluted context already forces you to (otherwise stop before the final answer).
Now generate the repair continuation.
[PROBLEM q]
{q}
[REASONING PREFIX c_0:alpha]
{prefix}
[CLEAN WINDOW w_clean -- for your reference only]
{w_clean}
[POLLUTED WINDOW w_poll -- this is what the agent saw at deployment]
{w_poll}
[YOUR OUTPUT: repair continuation that follows immediately after w_poll]'''

DIAGNOSIS = '''Question:
{q}
Student Solution:
{solution}
Your task involves three parts:
1. **Step-by-step Evaluation:** Go through the student solution carefully and identify key errors and potential misunderstandings that led to the incorrect solution.
2. **Final Judgement:** Provide an overall judgement on the correctness of the student's solution.
3. **First Error Step:** If the solution is incorrect, generate the step number where the first error occurs, otherwise generate N/A here
4. **Error Analysis:** If the solution is incorrect, analyse the cause and reasons for the first error step, otherwise generate N/A here
Here's the format I want:
Step-by-step Evaluation: [Provide a step by step examination of the student solution and identify key errors and misunderstandings here.]
Final Judgement: [Insert only **correct** or **wrong** here]
First Error Step: [Insert either N/A or the step number where the first error occurs]
Error Analysis: [Insert either N/A or the analysis of error in the first error among solution steps]
Please follow this format without any additional introductory or concluding statements.'''

REVISION = '''Question:
{q}
Your previous solution (incorrect):
{solution}
Task:
1) Identify the key mistake(s) in the previous solution.
2) Produce a corrected solution.
3) Provide the final answer in the form boxed{{answer}} at the end of your response.'''


class Prompts:
    def __init__(self, tokenizer, config):
        self.tok, self.cfg = tokenizer, config

    def chat(self, system, user):
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def fit(self, build, prefix):
        """Keep the episode's q, prefix and windows identical across role prompts."""
        ids = self.tok.encode(build(prefix), add_special_tokens=False)
        if len(ids) > self.cfg.prompt_tokens:
            raise ValueError(f"Role prompt has {len(ids)} tokens, exceeding prompt budget "
                             f"{self.cfg.prompt_tokens}; refusing to truncate the shared "
                             "reasoning prefix. An overflow policy needs an explicit "
                             "protocol decision; changing the cap departs from paper settings.")
        return ids

    def agent(self, q, prefix="", polluted=""):
        if self.cfg.tiny:
            return self.tok.encode(f"agent {q} prefix {prefix} polluted {polluted} answer")
        def build(p):
            base = self.chat(AGENT, q)
            if p or polluted:
                if not base.rstrip().endswith("<think>"):
                    base += "<think>\n"
                # A token boundary can be inside a word or number. Preserve the
                # sampled replacement verbatim; adding a separator changes it.
                base += p + polluted
            return base
        return self.fit(build, prefix)

    def polluter(self, q, prefix, clean):
        if self.cfg.tiny:
            return self.tok.encode(f"polluter {q} prefix {prefix} clean {clean} polluted")
        return self.fit(lambda p: self.chat("You are a helpful assistant.",
            POLLUTER.format(q=q, prefix=p, continuation=clean)), prefix)

    def repair(self, q, prefix, clean, polluted):
        if self.cfg.tiny:
            return self.tok.encode(f"repair {q} prefix {prefix} clean {clean} polluted {polluted}")
        return self.fit(lambda p: self.chat("You are a helpful reasoning assistant. Solve the user's problem.",
            REPAIR.format(q=q, prefix=p, w_clean=clean, w_poll=polluted)), prefix)
