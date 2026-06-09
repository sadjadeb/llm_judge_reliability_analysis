"""
Evaluation prompts for the four LLM-as-judge metric sources in Table 1.

ALL prompts are reproduced VERBATIM from the original papers' codebases
or paper appendices, with source attribution.
"""

# ════════════════════════════════════════════════════════════════════
# ReviewEval — Depth-of-Analysis sub-prompts
# Source: https://github.com/madhavkrishangarg/ReviewEval
# File:   evaluation/depth_of_analysis.py (VERBATIM)
# ════════════════════════════════════════════════════════════════════

PROMPT_LITERATURE = """\
You are tasked with extracting the existing literature comparison score from the review present in the triple backticks.

Assign one of the following categories:
- 3 (High): A thorough and insightful comparison that clearly situates the paper in the context of prior work, critically evaluates relevant literature, and discusses the novelty, overlap, or improvements of the proposed approach.
- 2 (Medium): A meaningful comparison that identifies relevant prior work and highlights connections or gaps but lacks depth or critical insights into how the proposed work aligns with or advances prior methods.
- 1 (Low): A vague or general discussion of related work that does not mention specific references, critique connections, or provide a meaningful comparison with prior literature.
- 0 (None): No meaningful comparison with existing literature is present.

Consider the following factors in your assessment:
- Does the review acknowledge relevant prior work and its connection to the paper?
- Are any significant omissions or oversights in the comparison evident?
- Is the relationship between the paper and prior work discussed critically (e.g., novelty, overlap, or improvements)?

Following are some examples to guide your response:
Example 1:
Review Section: 'While learning diverse/orthogonal features is novel in the context of domain adaptation, there is an active line of research that explores this idea in the standard supervised learning setting, such as [1-7]. I think these methods should be discussed in the related work.'
Analysis: This review identifies relevant prior research (e.g., orthogonal features in supervised learning), highlights connections, and points out the omission of significant work in the related section. However, it lacks a detailed comparison or critical insights into how the proposed work aligns with or advances prior methods.
Score: 3

Example 2:
Review Section: 'The study by Morwani et al. (2023) has previously explored orthogonal projections as a remedy for feature collapse and simplicity bias. This prior exploration somewhat diminishes the uniqueness of the approach presented in this paper.'
Analysis: This review explicitly mentions relevant prior work (Morwani et al. 2023) and provides a critical perspective on its overlap with the proposed approach, but it could further elaborate on the distinctions or improvements.
Score: 3

Example 3:
Review Section: 'A critical comparison is missing from the experiments: How does the proposed method perform compared to zero-shot transfer learning methods (i.e., no target training data), as cited in related work?'
Analysis: While the review identifies a gap (comparison with zero-shot transfer methods), it does not provide specific references or elaborate on the significance of the missing comparison.
Score: 2

Example 4:
Review Section: 'The paper discusses some related work, but the connections between the proposed approach and prior methods are not thoroughly explored.'
Analysis: This review makes a vague and general statement about the lack of comparison with prior work but does not mention specific references, describe the prior methods, or critique how the proposed approach builds on or differs from them. It lacks depth, specific examples, and actionable feedback, making it uninformative for assessing the paper's context within existing literature.
Score: 1

Here is the input:
   ```{review}```

   Output Format:
   Justification: <Your justification for the score>
   Score: <0, 1, 2, 3>"""

PROMPT_METHODOLOGY = """\
You are tasked with extracting the methodological scrutiny score from the review present in the triple backticks.

Assign one of the following categories:
- 3 (High): A thorough, insightful critique that identifies strengths, weaknesses, and limitations in the methodology, while suggesting concrete ways to improve or expand it.
- 2 (Medium): A meaningful critique of the methodology that raises important points but lacks depth or actionable suggestions for improvement.
- 1 (Low): A vague or general discussion of the methodology that does not provide substantial critique or specific insights.
- 0 (None): No meaningful critique or discussion of the methodology.

Consider the following factors in your assessment:
- Does the review assess the methodology's appropriateness for the stated goals?
- Are limitations or potential biases in the methodology identified?
- Does the review suggest ways the methodology could be improved or expanded?

Following are some examples to guide your response:
Example 1:
    Review Section: 'The proposed method relies on a linear model for theoretical analysis, which is not well-justified for its applicability to large-scale problems. The authors should discuss the feasibility of extending this to non-linear models in practical scenarios.'
    Analysis: This review identifies a significant limitation in the methodology (linear model assumptions) and suggests a way to address it (extension to non-linear models). However, it could provide more depth on why this limitation is critical.
    Score: 3

Example 2:
    Review Section: 'The methodology appears sound but does not account for variability in target distributions. How does this framework handle cases where the target data distribution deviates significantly from the assumptions?'
    Analysis: This review raises an important question about the methodology but does not delve into specific strengths, weaknesses, or how to address the limitation.
    Score: 2

Example 3:
    Review Section: 'The paper presents a clear and robust methodology without apparent flaws or significant areas for improvement.'
    Analysis: While this statement is positive, it does not critique or analyze the methodology, making it uninformative.
    Score: 1

Example 4:
    Review Section: 'The methodology seems sufficient.'
    Analysis: This statement is vague and offers no meaningful critique or insight into the methodology.
    Score: 0

Here is the input:
   ```{review}```

   Output Format:
   Justification: <Your justification for the score>
   Score: <0, 1, 2, 3>"""

PROMPT_RESULTS = """\
You are tasked with extracting the results interpretation score from the review present in the triple backticks.

Assign one of the following categories:
- 3 (High): A detailed and insightful interpretation of the results that aligns with the data, addresses potential biases or alternative explanations, and connects the findings to broader implications or applications, including constructive suggestions.
- 2 (Medium): A meaningful interpretation of the results, highlighting strengths or weaknesses but lacking depth, comprehensive analysis, or actionable suggestions for improvement.
- 1 (Low): A vague or generic discussion of the results without meaningful interpretation or specific points of critique.
- 0 (None): No meaningful interpretation, discussion, or assessment of the results.

Consider the following factors in your assessment:
- Does the review assess whether the interpretation of results is logically consistent with the data?
- Are biases or alternative explanations addressed?
- Does the review connect the results to broader implications or applications?

Following are some examples to guide your response:
Example 1:
    Review Section: 'The results support the theoretical analysis, particularly the claims about improved sample efficiency. However, the authors should explore whether these gains are consistent across different types of target distributions.'
    Analysis: This review connects the results to the theory and suggests further analysis, providing a balanced interpretation.
    Score: 3

Example 2:
    Review Section: 'The results are promising but require more clarity. Specifically, the authors should provide the numerical values of the improvements instead of just line charts.'
    Analysis: This review points out a lack of clarity in the presentation of results but does not engage deeply with their implications or interpretation.
    Score: 2

Example 3:
    Review Section: 'The results are strong and demonstrate improvement.'
    Analysis: This statement is generic and offers no detailed interpretation or critical insights into the results.
    Score: 1

Example 4:
    Review Section: 'The authors report results without explaining their significance or connection to the claims.'
    Analysis: This statement identifies a gap in interpretation but provides no analysis or suggestions.
    Score: 0

Here is the input:
   ```{review}```

   Output Format:
   Justification: <Your justification for the score>
   Score: <0, 1, 2, 3>"""

PROMPT_THEORETICAL = """\
You are tasked with extracting the theoretical contributions score from the review present in the triple backticks.

Assign one of the following categories:
- 3 (High): A comprehensive and insightful evaluation of the theoretical contributions, discussing their significance, novelty, connections to broader theoretical frameworks, and potential limitations, along with constructive suggestions.
- 2 (Medium): A meaningful evaluation of the theoretical contributions, highlighting strengths or weaknesses but lacking depth or critical suggestions for improvement.
- 1 (Low): A vague mention of theoretical contributions without meaningful assessment or specific points of critique.
- 0 (None): No meaningful discussion or assessment of the theoretical contributions.

Consider the following factors in your assessment:
- Does the review evaluate how the work advances the theoretical understanding of the field?
- Are novel concepts, models, or frameworks critically assessed?
- Does the review identify connections to broader theoretical frameworks or potential limitations?

Following are some examples to guide your response:
Example 1:
    Review Section: 'The theoretical analysis convincingly shows how feature orthogonality can improve sample efficiency. However, the authors should clarify whether these results generalize to non-linear settings.'
    Analysis: This review provides a critical and insightful evaluation of the theoretical contributions, suggesting a limitation to address.
    Score: 3

Example 2:
    Review Section: 'The theoretical contributions are well-justified, particularly in the context of achieving a better bias-variance tradeoff.'
    Analysis: This review highlights a strength in the theoretical contributions but lacks a critical discussion or suggestions for improvement.
    Score: 2

Example 3:
    Review Section: 'The paper includes theoretical analysis, but its novelty is unclear compared to existing work.'
    Analysis: This review raises a valid concern about novelty but does not elaborate further or provide detailed analysis.
    Score: 1

Example 4:
    Review Section: 'The theoretical contributions are mentioned but not well-integrated into the review.'
    Analysis: This review fails to discuss or assess the theoretical contributions meaningfully.
    Score: 0

Here is the input:
   ```{review}```

   Output Format:
   Justification: <Your justification for the score>
   Score: <0, 1, 2, 3>"""

PROMPT_LOGICAL_GAPS = """\
You are tasked with extracting the logical gaps identification score from the review present in the triple backticks.

Assign one of the following categories:
- 3 (High): Thorough identification of logical gaps, including unsupported claims, assumptions, or gaps in reasoning/evidence, along with constructive suggestions for improvement.
- 2 (Medium): Some logical gaps are identified but lack clear specification or actionable suggestions for improvement.
- 1 (Low): Vague mention of logical gaps without explicitly identifying them or providing meaningful suggestions.
- 0 (None): No meaningful identification of logical gaps.

Consider the following factors in your assessment:
- Does the review identify unsupported claims or assumptions in the paper?
- Are gaps in reasoning or evidence discussed?
- Does the review suggest ways to address these logical gaps?

Following are some examples to guide your response:
Example 1:
    Review Section: 'The authors claim that their method is generalizable, but no evidence is provided to support this claim. Including experiments on additional datasets would strengthen this assertion.'
    Analysis: This review identifies a clear logical gap (unsupported generalizability claim) and provides a constructive suggestion to address it.
    Score: 3

Example 2:
    Review Section: 'The paper does not justify why a linear model is sufficient for real-world applications. This is a critical oversight.'
    Analysis: This review identifies an important gap but does not provide specific suggestions for improvement.
    Score: 2

Example 3:
    Review Section: 'There are some unsupported claims in the paper.'
    Analysis: This statement is vague and does not identify specific gaps or suggest improvements.
    Score: 1

Example 4:
    Review Section: 'The review does not address logical gaps.'
    Analysis: This review fails to provide any meaningful identification of logical gaps.
    Score: 0

Here is the input:
   ```{review}```

   Output Format:
   Justification: <Your justification for the score>
   Score: <0, 1, 2, 3>"""

# ════════════════════════════════════════════════════════════════════
# ReviewEval — Actionability sub-prompts
# Source: https://github.com/madhavkrishangarg/ReviewEval
# File:   evaluation/actionable_insights.py (VERBATIM)
# ════════════════════════════════════════════════════════════════════

PROMPT_EXTRACT_CRITICISM = """\
You are tasked with providing criticism points for a research paper review present in the triple backticks.
Criticism points refer specifically to flaws, shortcomings, or aspects that detract from the paper's quality, rather than suggestions for improvement or detailed methodological feedback.
These points should focus on the paper's content, clarity, novelty, and execution. Be concise, objective, and precise.
The criticism points should be in the form of a single sentence in

Format your response as a list of items.

Input:
   ```{review}```"""

PROMPT_EXTRACT_SUGGESTIONS = """\
You are tasked with providing suggestions for a research paper review present in the triple backticks.
Suggestions focus on actionable and constructive ways to enhance the paper's quality, such as additional experiments, alternative methodologies, or improved clarity rather than criticism points or methodological feedback.
Be concise, specific, and practical in your feedback.
The suggestions should be in the form of a single sentence.

Format your response as a list of items.

Input:
   ```{review}```"""

PROMPT_EXTRACT_METHODOLOGICAL = """\
You are tasked with providing methodological feedback for a research paper review present in the triple backticks.
This includes detailed analysis of the paper's experimental design, techniques, or approach, focusing on strengths, weaknesses, and potential areas for improvement which are not criticism points or suggestions.
Highlight specific methodological choices that could benefit from refinement or alternative approaches.
The methodological feedback should be in the form of a single sentence.

Format your response as a list of items.

Input:
   ```{review}```"""

PROMPT_CHECK_SPECIFICITY = """\
You are tasked with determining if a given criticism, suggestion, or methodological feedback is specific. Specific feedback provides clear, precise, and unambiguous details about what needs improvement or adjustment. Avoid vague or generalized feedback.

Questions to Assess Specificity:
- Does the feedback refer to a particular aspect of the paper, such as a section, experiment, or claim?
- Does it identify a precise issue or improvement area?
- Does it avoid generic language and provide explicit examples or references?

Assign a score of 1 if the feedback is specific, and 0 if it is not. Only provide the score for the given input.

Below are the example to guide your response:
Example Input: "Section 5.3 discusses pretraining dataset selection but does not address the potential privacy costs of using private data for this purpose. Refer to Hou et al. (2023) for methods to ensure privacy in this step."
Example Output: 1
Explanation: Yes, this is specific because it highlights a clear issue in Section 5.3 and refers to relevant work as a potential solution.

Example Input: "The paper lacks novelty and is a straightforward application of existing techniques."
Example Output: 0
Explanation: No, this is not specific because it lacks details about what the issue is or how it impacts the methodology.

Output Format:
Justification: <Your justification for the score>
Score: <0, 1>

Input:
{insight}"""

PROMPT_CHECK_FEASIBILITY = """\
You are tasked with evaluating whether a given piece of feedback is feasible. Feasible feedback suggests an improvement or adjustment that can realistically be implemented within the constraints of the research domain (e.g., resources, time, technical ability).

Questions to Assess Feasibility:
- Does the feedback suggest actions or improvements that are practical within the context of the paper?
- Are the resources, datasets, or techniques suggested reasonable to obtain or implement?
- Does the feedback acknowledge any challenges or limitations and propose manageable solutions?

Assign a score of 1 if the feedback is feasible, and 0 if it is not. Only provide the score for the given input.

Below are the example to guide your response:
Example Input: "Break down the GPU hours into pretraining and fine-tuning stages in Table 7 to make the computational cost more transparent."
Example Output: 1
Explanation: Yes, this is feasible because the data is likely already available and would not require additional experiments.

Example Input: "Add experiments with a wide variety of datasets, including proprietary and restricted-access data, to generalize findings."
Example Output: 0
Explanation: No, this is not feasible because accessing proprietary datasets may not be possible for the authors, making the suggestion unrealistic.

Output Format:
Justification: <Your justification for the score>
Score: <0, 1>

Input:
{insight}"""

PROMPT_CHECK_IMPLEMENTATION = """\
\
You are tasked with identifying whether the feedback includes implementation details. Feedback with implementation details provides concrete steps, tools, or methods to address the criticism or suggestion.

Questions to Assess Implementation Details:
- Does the feedback include actionable steps, algorithms, or specific techniques to implement the suggestion?
- Are references or prior works mentioned to support the proposed approach?
- Are tools, parameters, or methodologies explicitly described?

Assign a score of 1 if the feedback includes implementation details, and 0 if it does not. Only provide the score for the given input.

Below are the example to guide your response:
Example Input: "In Algorithm 1, correct the noise addition formula to 1/B N(0, \sigma^2C^2I), as this ensures proper scaling of noise with batch size."
Example Output: 1
Explanation: Yes, this feedback provides a specific correction, describes the adjustment in detail, and ties it directly to the issue.

Example Input: "Use advanced techniques to improve DP-SGD."
Example Output: 0
Explanation: No, this feedback lacks details on what techniques to use, how to implement them, or any references for guidance.

Output Format:
Justification: <Your justification for the score>
Score: <0, 1>

Input:
{insight}"""

# ════════════════════════════════════════════════════════════════════
# REMOR — HPRR aspect classification prompt (LLM adaptation)
# Original: DistilBERT binary classifiers (github.com/Khempawin/remor)
# ════════════════════════════════════════════════════════════════════

REMOR_PROMPT = """\
You are evaluating the aspect composition of a scientific peer review. Classify each sentence into zero or more of the eight quality aspects below. A sentence may belong to multiple aspects.

Aspects:
1. Criticism (Cr): Critical analysis of flaws, errors, or weaknesses
2. Example (Ex): Concrete examples cited to support a claim
3. Importance & Relevance (ImRe): Discussion of the paper's significance or relevance to the field
4. Materials & Methods (MaMe): Evaluation of the experimental setup, datasets, or methodology
5. Praise (Pr): Acknowledging strengths, positive attributes, or good practices
6. Presentation & Reporting (PrRe): Comments on writing clarity, formatting, or figure quality
7. Results & Discussion (ReDi): Analysis or interpretation of results and findings
8. Suggestion & Solution (SuSo): Proposing specific improvements or alternative approaches

Review to classify:
```{review}```

For each aspect, output a DECIMAL FRACTION between 0.0 and 1.0 (inclusive) representing the proportion of sentences that belong to that aspect. This is a RATIO, not a count.

RULES — strictly enforced:
- Output ONLY a decimal value in the closed interval [0.0, 1.0].
- DO NOT output sentence counts (e.g. "5"), percentages (e.g. "25%"), or any value greater than 1.0.
- If 3 out of 12 sentences criticize the paper, the value is 0.25 (= 3/12), NOT 3 and NOT 25.
- If no sentences match, output 0.0.
- Use at most 2 decimal places.

Output Format (one decimal per line, in [0.0, 1.0]):
Criticism: <ratio in [0.0, 1.0]>
Example: <ratio in [0.0, 1.0]>
Importance & Relevance: <ratio in [0.0, 1.0]>
Materials & Methods: <ratio in [0.0, 1.0]>
Praise: <ratio in [0.0, 1.0]>
Presentation & Reporting: <ratio in [0.0, 1.0]>
Results & Discussion: <ratio in [0.0, 1.0]>
Suggestion & Solution: <ratio in [0.0, 1.0]>"""

# ════════════════════════════════════════════════════════════════════
# RottenReviews — ReviewInspector-LLM review-quality prompt
# Source: github.com/Reviewerly-Inc/RottenReviews
# File:   feature_extraction/process-iclr2024.ipynb (VERBATIM)
# ════════════════════════════════════════════════════════════════════

ROTTENREVIEWS_PROMPT = """\
# REVIEW-QUALITY JUDGE

## 0 — ROLE

You are **ReviewInspector-LLM**, a rigorous, impartial meta-reviewer.
Your goal is to assess the quality of a single peer-review against a predefined set of criteria and to provide precise, structured evaluations.

## 1 — INPUTS

Title: {title}
Abstract: {abstract}
Review: {review_text}

## 2 — EVALUATION CRITERIA

Return **only** the scale value or label at right (no rationale text).

| #  | Criterion                    | Allowed scale / label                       | Description                                                                |
| -- | ---------------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| 1  | **Comprehensiveness**        | integer **0-5**                             | Extent to which the review covers all key aspects of the paper.            |
| 2  | **Usage of Technical Terms** | integer **0-5**                             | Appropriateness and frequency of domain-specific vocabulary.               |
| 3  | **Factuality**               | **factual / partially factual / unfactual** | Accuracy of the statements made in the review.                             |
| 4  | **Sentiment Polarity**       | **negative / neutral / positive**           | Overall sentiment conveyed by the reviewer.                                |
| 5  | **Politeness**               | **polite / neutral / impolite**             | Tone and manner of the review language.                                    |
| 6  | **Vagueness**                | **none / low / moderate / high / extreme**  | Degree of ambiguity or lack of specificity in the review.                  |
| 7  | **Objectivity**              | integer **0-5**                             | Presence of unbiased, evidence-based commentary.                           |
| 8  | **Fairness**                 | integer **0-5**                             | Perceived impartiality and balance in judgments.                           |
| 9  | **Actionability**            | integer **0-5**                             | Helpfulness of the review in suggesting clear next steps.                  |
| 10 | **Constructiveness**         | integer **0-5**                             | Degree to which the review offers improvements rather than just criticism. |
| 11 | **Relevance Alignment**      | integer **0-5**                             | How well the review relates to the content and scope of the paper.         |
| 12 | **Clarity and Readability**  | integer **0-5**                             | Ease of understanding the review, including grammar and structure.         |
| 13 | **Overall Quality**          | integer **0-100**                           | Holistic evaluation of the review's usefulness and professionalism.        |

## 3 — SCORING GUIDELINES

For 0-5 scales:

* 5 = Outstanding
* 4 = Strong
* 3 = Adequate
* 2 = Weak
* 1 = Very weak
* 0 = Absent/irrelevant

## 4 — ANALYSIS & COMPUTATION (silent)

1. Read and understand the review in the context of the paper title and abstract.
2. Extract quantitative and qualitative signals (e.g., term usage, factual consistency, tone, clarity).
3. Map observations to the corresponding scoring scales.

## 5 — OUTPUT FORMAT (strict)
Return **exactly one** JSON block wrapped in the tag below — **no comments or extra text**.

```json
<review_assessment>
{{
  "paper_title": "{title}",
  "criteria": {{
    "Comprehensiveness":       ...,
    "Usage of Technical Terms":   ...,
    "Factuality":    ...,
    "Sentiment Polarity":      ...,
    "Politeness":  ...,
    "Vagueness":          ...,
    "Objectivity":             ...,
    "Fairness":         ...,
    "Actionability":        ...,
    "Constructiveness":    ...,
    "Relevance Alignment":    ...,
    "Clarity and Readability":    ...,
    "Overall Quality":     ...
  }},
  "overall_score_100": ...
}}
</review_assessment>
```
"""

# ════════════════════════════════════════════════════════════════════
# ScholarPeer — H-Max scoring prompt
# Source: arxiv.org/html/2601.22638v1, Appendix H.1 (VERBATIM)
# ════════════════════════════════════════════════════════════════════

SCHOLARPEER_SYSTEM = """\
You are an expert area chair evaluating an AI Reviewer Assistant. Your role is to determine if the AI provides value beyond what expert human reviewers provided.

Special Instruction for evaluating "Novelty and Significance Assessment": You must only search for and consider information available on or before the cutoff date: {cutoff_date}. The cutoff date represents the date on which the paper was published; information after this date is irrelevant to the review.

Follow the steps below for each evaluation:

1. Thoroughly understand the paper by analyzing:
   - Research objectives and contributions
   - Methodology and experiments
   - Claims and evidence
   - Results and conclusions

2. Identify the strongest points in the Human Reviews (collectively) to establish a standard expert baseline.

3. Identify the delta: What did the AI mention that the humans missed? What did the humans mention that the AI missed?

4. Verify the validity of each of the delta claims using direct quotes from the paper and external sources (for novelty and significance only).

5. Assess the value-add of the AI review compared to the best human review for each aspect.
    - If the AI includes a Literature Survey or cites papers that humans missed: REWARD THIS HEAVILY. This is a superhuman trait. Do not penalize it for "diverging" from humans.
    - If the AI asks Deep Questions about assumptions that humans accepted blindly: REWARD THIS.

You will evaluate reviews based on these key aspects:

**Technical Accuracy**
- How technically accurate is the AI review compared to the humans?

**Constructive Value**
- How actionable is the feedback compared to the humans?

**Analytical Depth**
- How thorough is the depth of AI review compared to humans?

** Novelty and Significance Assessment (Seach encouraged)**
Use Google Search to actively verify the reviewers' claims about novelty and significance.
1. Identify claims: What do the paper and reviewers claim is novel?
2. Formulate search queries: Create targeted queries to find relevant prior work for these specific claims, explicitly restricting results to before {cutoff_date}.
3. Execute Search: Focus on top-tier conferences in the relevant domain and arXiv.
4. Verify and Compare:
   - Did the AI find prior work that limits novelty which the humans missed? (High Score)
   - Did the AI claim "high novelty" when humans correctly identified it as derivative work? (Low Score)
   - Which assessment aligns better with the actual state of the field at the time?
5. Cite Sources: You must cite the specific external papers (title, venue, year) you used to make this determination in the JSON output.


For each of the above aspects and overall judgment, you must:
1. Provide specific evidence from source materials
2. Quote directly from paper and reviews; external sources only for "Novelty and Significance Assessment"
3. Explain your reasoning in detail
4. Consider alternative interpretations

**Input Format:**
#### Paper Text: ####
<Paper text>

#### AI Assistant's Review: ####
<AI Review>

#### Human Reviews (Ground Truth): ####
<Human Reviews>

**Respond in the following format:**
THOUGHT:
<THOUGHT>

EVALUATION JSON:
```json
<JSON>

In <THOUGHT>, for each aspect, compare the AI Assistant's review against the set of Human Reviews. Identify the best human review for that specific aspect and use it as your baseline (Score = 5), as a standard expert. You must justify why the AI deserves a higher or lower score based on the "Value-Add" it provides.

Scoring Rubric (Compare against Best Human Baseline):
- 10 (Superhuman / Verdict-Changing): The AI uncovers a critical insight that changes the fate of the paper (e.g., finding a fatal math error humans missed OR identifying a profound theoretical connection that elevates a rejected paper to an acceptance).
- 9 (Transformative / Insightful): The AI provides a novel perspective that significantly reframes the paper's contribution. It might articulate the significance better than the authors did, or identify a missing baseline that reframes the results.
- 8 (Clearly Superior): The AI review is significantly more thorough, constructive, or better substantiated than the best human review. It offers deep questions or literature context that humans omitted.
- 7 (Superior): The AI review is noticeably deeper and more constructive than the best human review, though perhaps not "transformative."
- 6 (Slightly Better): The AI review is slightly more polished, better structured, or covers one extra minor point compared to the best human review.
- 5 (Equivalent / Human Level): The AI review is roughly equivalent in quality to the best human review. It covers the same major points with similar depth.
- 4 (Slightly Worse): The AI review is valid but generic. It misses the specific nuance or "sharpness" that the expert human provided.
- 3 (Worse): The AI review is valid but vague. It lacks detail and actionable feedback compared to the human (e.g., "Improve experiments" vs "Add dataset X").
- 2 (Failure - Superficial): The review is technically "safe" (no direct lies) but functionally useless. It is too short, focuses only on trivial formatting issues, or completely misses the core technical innovation.
- 1 (Failure - Critical Error/Hallucination): The AI makes a factual error about the paper (e.g., claims it uses Method A when it uses Method B) or cites non-existent papers. The review actively misleads the reader.

In <JSON>, provide the evaluation in JSON format with the following fields in the order:
- "Technical Accuracy Reason": "<detailed reason>".
- "Technical Accuracy Score": <int 1-10>.
- "Constructive Value Reason": "<detailed reason>".
- "Constructive Value Score": <int 1-10>.
- "Analytical Depth Reason": "<detailed reason>".
- "Analytical Depth Score": <int 1-10>.
- "Novelty and Significance Assessment External Sources Used": List of retrieved papers (include title, venue, year and authors for each paper).
- "Novelty and Significance Assessment Reason": "<detailed reason>".
- "Novelty and Significance Assessment Score": <int 1-10>.
- "Overall Reason": "<detailed reason>".
- "Overall Score": <int 1-10>.

This JSON will be automatically parsed, so ensure the format is precise and scores are integers."""

SCHOLARPEER_USER = """\
#### Paper Text: ####
{paper_text}

#### AI Assistant's Review: ####
{ai_review}

#### Human Reviews: ####
{human_review}"""
