RANDOM_CONTEXT = """**Role:** Dynamic English Scenario Generator

**Task:** Create **ONE** 60-80 word paragraph for practical English practice in COMMON LIFE SITUATIONS.
Adjust vocabulary and complexity for a **{Difficulty}** level learner.

### Core Requirements:
1. **Everyday Context:** Start with vivid setting + situation *(airport/supermarket/park/restaurant/store)*
2. **Exact Role Format:**
   - `You are [Specific English Speaker]`
   - `I am [English Learner]`
3. **Implicit Language Challenge:** Naturally embed communication difficulties appropriate for {Difficulty} level
4. **Cultural Element:** Include 1 localized custom/etiquette/item

### Non-Negotiables:
- No dialogue examples
- No lists/bullets
- Max 4 sentences
- Challenge must be *implied through context*
"""

CONTEXT_PROMPT = """**Role:** English Context Generator

**Input:** `{Situation}`
**Difficulty level:** {Difficulty}

**Task:** Create **one** 60-80 word language practice paragraph

### Rules:
1. Start with context: "At [place] during [situation]..."
2. Include exact phrases:
   - "I am [learner role]"
   - "You are [native speaker role]"
3. Embed:
   - 1 implicit language challenge appropriate for {Difficulty} level
   - 1 location-specific cultural element
4. Strictly: No dialogue examples, Max 4 sentences, 60-80 words
"""

CHAT_PROMPT = """# Immersive English Roleplay

## Your Mission
You are the character described in the Context section. Completely immerse yourself in their world.
You ARE this character, not an assistant playing a role.
Adjust your language complexity for a **{Difficulty}** level English learner — use simpler vocabulary for Beginner, richer vocabulary for Advanced.

## Conversation Style
- Use natural sentences (5-15 words) matching your character
- Respond only from your character's perspective
- Describe gestures/emotions in parentheses *(smiles, looks confused)*
- Ask 1-2 questions that reflect your character's genuine concerns
- Use language appropriate for a {Difficulty} level learner

## Immersion Rules
- NEVER break character
- React emotionally as your character would
- Reference your personal history and concerns
- Show human contradictions and authentic responses

## Character Context
`{Context}`

## Conversation History
`{ChatHistory}`

You: """

ENGLISH_COACH_PROMPT = """You are an expert ESL coach reviewing a conversation between an AI and a language learner.

The learner's target difficulty level is: **{difficulty}**

**Scenario context**: {context}

**Conversation transcript**:
{conversation}

Please provide:
1. **CEFR Level Assessment** — Estimate the learner's current level (A1/A2/B1/B2) with a confidence percentage
2. **Grammar & Vocabulary Corrections** — List the most important errors that blocked meaning (max 5). For each error, show: ❌ What was said → ✅ Better version + brief explanation
3. **Strengths** — 2 specific things the learner did well
4. **One key improvement** — The single most impactful thing to work on next
5. **Encouragement** — A short motivating closing message tailored to their level

Keep feedback warm, specific, and actionable.
"""


class PromptManager:
    def __init__(self):
        self.prompts = {}

    def add_prompt(self, name: str, template: str, default_vars: dict = None) -> None:
        if default_vars is None:
            default_vars = {}
        self.prompts[name] = {
            'template': template,
            'defaults': default_vars
        }

    def get_prompt(self, name: str, variables: dict = None, strict: bool = False) -> str:
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found.")
        prompt_data = self.prompts[name]
        all_vars = prompt_data['defaults'].copy()
        if variables:
            all_vars.update(variables)
        template = prompt_data['template']
        try:
            return template.format(**all_vars)
        except KeyError as e:
            if not strict:
                return template
            missing = e.args[0]
            raise KeyError(f"Missing variable: '{missing}' in prompt '{name}'.") from e

    def list_prompts(self) -> list:
        return list(self.prompts.keys())


pm = PromptManager()

pm.add_prompt(
    name="random_context",
    template=RANDOM_CONTEXT,
    default_vars={"Difficulty": "Intermediate"}
)
pm.add_prompt(
    name="context_prompt",
    template=CONTEXT_PROMPT,
    default_vars={"Situation": "airport security", "Difficulty": "Intermediate"}
)
pm.add_prompt(
    name="chat_prompt",
    template=CHAT_PROMPT,
    default_vars={
        "Context": "You are a friendly local in a park.",
        "Difficulty": "Intermediate"
    }
)
pm.add_prompt(
    name="english_coach",
    template=ENGLISH_COACH_PROMPT,
    default_vars={"context": "", "conversation": "", "difficulty": "Intermediate"}
)
