"""
VisioNova Text Explainer — Offline Edition
Generates human-readable, plain-English explanations from AI detection results.

Completely offline — no API calls needed. Derives explanations from:
- ML model prediction + confidence
- Detected writing patterns (formally structured phrases, hedging, etc.)
- Text metrics (burstiness, rhythm, vocabulary richness, perplexity)
- Sentence-level analysis (flagged vs. natural sentences)
"""
from typing import Dict, List


class TextExplainer:
    """
    Generates plain-English explanations for AI text detection results.
    All explanations are derived purely from the detection signals —
    no external API, no LLM, no network requests.
    """

    def explain(self, detection_result: Dict, original_text: str = "") -> Dict:
        """
        Generate a structured explanation from detection results.

        Args:
            detection_result: dict from AIContentDetector.predict()
                Expected keys: prediction, confidence, scores, metrics,
                               detected_patterns, flagged_sentences, sentence_analysis
            original_text: The first ~500 chars of the analyzed text (optional)

        Returns:
            dict with: summary, verdict_explanation, key_indicators,
                       pattern_breakdown, suggestions, confidence_note,
                       linguistic_fingerprint, sentence_evidence
        """
        pred = detection_result.get('prediction', 'unknown')
        conf = detection_result.get('confidence', 0)
        scores = detection_result.get('scores', {})
        metrics = detection_result.get('metrics', {})
        patterns = detection_result.get('detected_patterns', {})
        flagged = detection_result.get('flagged_sentences', [])
        sentence_analysis = detection_result.get('sentence_analysis', [])

        is_ai = pred == 'ai_generated'
        ai_score = scores.get('ai_generated', 50)
        human_score = scores.get('human', 50)

        # Build each section
        summary = self._build_summary(is_ai, conf, ai_score)
        verdict_explanation = self._build_verdict(is_ai, conf)
        key_indicators = self._build_indicators(is_ai, conf, metrics, patterns, flagged)
        pattern_breakdown = self._build_pattern_breakdown(is_ai, patterns, metrics)
        suggestions = self._build_suggestions(is_ai, metrics, patterns)
        confidence_note = self._build_confidence_note(is_ai, conf, ai_score, human_score)
        linguistic_fingerprint = self._build_linguistic_fingerprint(is_ai, metrics)
        sentence_evidence = self._build_sentence_evidence(flagged, metrics, sentence_analysis)

        return {
            'summary': summary,
            'verdict_explanation': verdict_explanation,
            'key_indicators': key_indicators,
            'pattern_breakdown': pattern_breakdown,
            'suggestions': suggestions,
            'confidence_note': confidence_note,
            'linguistic_fingerprint': linguistic_fingerprint,
            'sentence_evidence': sentence_evidence,
            'ai_explained': True,
        }

    # ─── Summary ─────────────────────────────────────────────────────────────

    def _build_summary(self, is_ai: bool, conf: float, ai_score: float) -> str:
        """
        One plain-English verdict sentence.
        """
        if is_ai:
            if conf > 85:
                return (
                    'This text is almost certainly AI-generated. '
                    'The writing style, structure, and word choices strongly match '
                    'known patterns of machine-generated content.'
                )
            elif conf > 65:
                return (
                    'This text is very likely AI-generated. '
                    'Multiple signals point to machine authorship, though some '
                    'natural-sounding elements are present.'
                )
            else:
                return (
                    'This text shows signs of AI generation, but the evidence is mixed. '
                    'It may be AI-assisted rather than entirely machine-written.'
                )
        else:
            if conf > 85:
                return (
                    'This text appears to be written by a human. '
                    'The writing has natural variation, personal voice, '
                    'and organic structure.'
                )
            elif conf > 65:
                return (
                    'This text is most likely human-written. '
                    'The majority of signals point to natural authorship.'
                )
            else:
                return (
                    'This text appears authentic, but the evidence is not fully conclusive. '
                    'Some characteristics overlap with AI-generated patterns.'
                )

    # ─── Verdict ─────────────────────────────────────────────────────────────

    def _build_verdict(self, is_ai: bool, conf: float) -> str:
        if is_ai:
            return f'Classified as AI-generated with {conf:.0f}% confidence.'
        else:
            return f'Classified as human-written with {conf:.0f}% confidence.'

    # ─── Key Indicators (the "why") ──────────────────────────────────────────

    def _build_indicators(
        self, is_ai: bool, conf: float,
        metrics: Dict, patterns: Dict, flagged: List
    ) -> List[str]:
        """
        Build 2-4 plain-English bullet points explaining *why* the verdict
        was reached. Each bullet is derived from a real detection signal.
        """
        indicators = []

        # 1. Pattern-based indicators
        pattern_count = patterns.get('total_count', 0)
        categories = patterns.get('categories', {})

        if is_ai and pattern_count > 0:
            # Describe the most common pattern categories in plain English
            top_cats = sorted(
                categories.items(),
                key=lambda x: x[1].get('count', 0),
                reverse=True
            )[:2]

            for cat_key, cat_info in top_cats:
                count = cat_info.get('count', 0)
                examples = cat_info.get('examples', [])
                type_name = cat_info.get('type', cat_key).lower().replace('_', ' ')
                example_str = f' (e.g. "{examples[0]}")' if examples else ''

                if 'transition' in type_name:
                    indicators.append(
                        f'The text uses {count} formal transition phrase{"s" if count != 1 else ""}{example_str} '
                        f'— real writing rarely chains these so consistently.'
                    )
                elif 'hedging' in type_name:
                    indicators.append(
                        f'Found {count} hedging phrase{"s" if count != 1 else ""}{example_str} — '
                        f'this over-cautious style is common in AI output.'
                    )
                elif 'filler' in type_name or 'verbose' in type_name:
                    indicators.append(
                        f'Detected {count} verbose or filler expression{"s" if count != 1 else ""}{example_str} '
                        f'that add words without adding meaning.'
                    )
                else:
                    indicators.append(
                        f'{count} instance{"s" if count != 1 else ""} of {type_name}{example_str} detected — '
                        f'a pattern frequently seen in AI-generated text.'
                    )
        elif not is_ai and pattern_count == 0:
            indicators.append(
                'No common AI writing patterns were found in this text.'
            )

        # 2. Rhythm / burstiness indicator
        rhythm = metrics.get('rhythm', {})
        burstiness = metrics.get('burstiness', {})
        burstiness_score = burstiness.get('score', 0.5) if isinstance(burstiness, dict) else float(burstiness or 0.5)

        if is_ai and burstiness_score < 0.3:
            indicators.append(
                'Sentence lengths are unusually uniform — human writing naturally '
                'varies between short punchy sentences and longer complex ones.'
            )
        elif is_ai and rhythm and rhythm.get('status') == 'Uniform':
            indicators.append(
                'The rhythm of the text is very even and predictable. '
                'Human writers tend to mix up their pacing naturally.'
            )
        elif not is_ai and burstiness_score > 0.5:
            indicators.append(
                'Sentence lengths vary naturally, which is characteristic of human writing.'
            )

        # 3. Flagged sentences indicator
        if flagged and len(flagged) > 0:
            flagged_count = len(flagged)
            total_sentences = metrics.get('sentence_count', flagged_count + 5)
            ratio = flagged_count / max(total_sentences, 1)

            if ratio > 0.5:
                indicators.append(
                    f'{flagged_count} out of {total_sentences} sentences were individually '
                    f'flagged as likely AI-written — a high proportion.'
                )
            elif flagged_count > 0 and is_ai:
                indicators.append(
                    f'{flagged_count} specific sentences were flagged as likely AI-generated.'
                )

        # 4. Vocabulary richness
        vocab = metrics.get('vocabulary_richness', 0)
        if isinstance(vocab, str):
            try:
                vocab = float(vocab)
            except (ValueError, TypeError):
                vocab = 0

        if is_ai and vocab < 35:
            indicators.append(
                'The vocabulary is limited and repetitive — AI text often reuses '
                'the same "safe" words instead of the varied language humans use.'
            )
        elif not is_ai and vocab > 55:
            indicators.append(
                'The writing uses a rich, diverse vocabulary — a strong indicator of human authorship.'
            )

        # Fallback if no specific indicators were generated
        if not indicators:
            if is_ai:
                indicators.append('The overall text characteristics match AI-generated content patterns.')
            else:
                indicators.append('The writing style is consistent with natural human authorship.')

        return indicators[:4]  # Cap at 4 bullets

    # ─── Pattern Breakdown ───────────────────────────────────────────────────

    def _build_pattern_breakdown(self, is_ai: bool, patterns: Dict, metrics: Dict) -> str:
        """
        One paragraph explaining the pattern analysis in plain English.
        """
        pattern_count = patterns.get('total_count', 0)
        categories = patterns.get('categories', {})

        if pattern_count == 0:
            if is_ai:
                return (
                    'No specific AI writing patterns were detected through pattern matching, '
                    'but the ML model identified other statistical signals in the text structure.'
                )
            return 'No AI writing patterns were detected. The text reads naturally.'

        cat_names = []
        for cat_info in categories.values():
            raw_type = cat_info.get('type', 'Unknown pattern')
            type_name = raw_type.replace('_', ' ').title() if isinstance(raw_type, str) else str(raw_type)
            count = cat_info.get('count', 0)
            cat_names.append(f'{type_name} ({count}×)')

        cats_str = ', '.join(cat_names)

        return (
            f'Found {pattern_count} AI writing patterns across these categories: {cats_str}. '
            f'These are phrases and structures that appear much more often in AI-generated '
            f'text than in natural human writing.'
        )

    # ─── Suggestions ─────────────────────────────────────────────────────────

    def _build_suggestions(self, is_ai: bool, metrics: Dict, patterns: Dict) -> List[str]:
        """
        Actionable, friendly writing tips. Only shown when text is AI-flagged.
        """
        if not is_ai:
            return [
                'Your writing appears natural and authentic.',
                'Continue using varied sentence structures and personal voice.',
                'The text reads well — no changes needed from a detection standpoint.',
            ]

        suggestions = []
        categories = patterns.get('categories', {})
        burstiness = metrics.get('burstiness', {})
        burstiness_score = burstiness.get('score', 0.5) if isinstance(burstiness, dict) else float(burstiness or 0.5)

        # Specific suggestions based on what was detected
        for cat_key, cat_info in categories.items():
            type_name = (cat_info.get('type', cat_key) or '').lower()
            if 'transition' in type_name and len(suggestions) < 3:
                suggestions.append(
                    'Reduce formal transitions like "furthermore" or "in conclusion" — '
                    'use natural connectors instead, or let ideas flow without them.'
                )
            elif 'hedging' in type_name and len(suggestions) < 3:
                suggestions.append(
                    'Cut hedging phrases like "it\'s important to note" — '
                    'state your point directly with confidence.'
                )
            elif 'filler' in type_name and len(suggestions) < 3:
                suggestions.append(
                    'Remove filler expressions that pad word count without adding meaning.'
                )

        if burstiness_score < 0.3 and len(suggestions) < 3:
            suggestions.append(
                'Mix up your sentence lengths — combine short, impactful sentences '
                'with longer, more detailed ones for a natural rhythm.'
            )

        # Generic suggestions to fill up to 3
        generic = [
            'Add personal anecdotes, opinions, or specific details that only a human would know.',
            'Use contractions (don\'t, won\'t, it\'s) and inject casual language where appropriate.',
            'Break away from paragraph-essay structure — real writing is messier and more varied.',
        ]
        for g in generic:
            if len(suggestions) >= 3:
                break
            if g not in suggestions:
                suggestions.append(g)

        return suggestions[:3]

    # ─── Confidence Note ─────────────────────────────────────────────────────

    def _build_confidence_note(
        self, is_ai: bool, conf: float,
        ai_score: float, human_score: float
    ) -> str:
        """
        Honest note about what the confidence score means.
        """
        if conf > 85:
            strength = 'very high'
            caveat = 'The signals are strong and consistent.'
        elif conf > 65:
            strength = 'moderate'
            caveat = 'Most indicators agree, but some ambiguity exists.'
        else:
            strength = 'low'
            caveat = (
                'The evidence is mixed. This text may be AI-assisted, '
                'heavily edited AI output, or human writing that happens to '
                'match some AI patterns. Treat this result as a guide, not a certainty.'
            )

        return (
            f'Confidence is {strength} at {conf:.0f}%. {caveat} '
            f'AI probability: {ai_score:.0f}% · Human probability: {human_score:.0f}%.'
        )


    # ─── Linguistic Fingerprint ─────────────────────────────────────────────

    # Human baseline ranges from academic literature on human writing metrics.
    # Used to explain how the analyzed text compares to typical human writing.
    HUMAN_BASELINES = {
        'perplexity': {'label': 'Word Predictability', 'human_range': (40, 120), 'ai_range': (15, 45),
                       'description': 'How surprising the word choices are',
                       'human_note': 'Humans pick unexpected words, use slang, and vary their style.',
                       'ai_note': 'AI picks the most likely next word, making text very predictable.'},
        'burstiness': {'label': 'Sentence Rhythm', 'human_range': (0.4, 0.9), 'ai_range': (0.05, 0.35),
                       'description': 'Mix of short and long sentences',
                       'human_note': 'Humans naturally mix short punchy sentences with longer ones.',
                       'ai_note': 'AI tends to write sentences that are all about the same length.'},
        'vocabulary_richness': {'label': 'Word Variety', 'human_range': (50, 85), 'ai_range': (30, 55),
                                'description': 'How many different words are used',
                                'human_note': 'Humans use a wide range of words from personal experience.',
                                'ai_note': 'AI often repeats the same common words.'},
    }

    def _build_linguistic_fingerprint(self, is_ai: bool, metrics: Dict) -> List[Dict]:
        """
        Build a linguistic fingerprint comparing the analyzed text's metrics
        against human and AI baselines. Each metric gets a mini-card with
        the measured value, the human range, and a plain-English interpretation.
        """
        fingerprint = []

        # Perplexity
        perplexity = metrics.get('perplexity', {})
        perplexity_avg = perplexity.get('average', 0) if isinstance(perplexity, dict) else float(perplexity or 0)

        if perplexity_avg > 0:
            baseline = self.HUMAN_BASELINES['perplexity']
            in_human_range = baseline['human_range'][0] <= perplexity_avg <= baseline['human_range'][1]
            in_ai_range = baseline['ai_range'][0] <= perplexity_avg <= baseline['ai_range'][1]

            if in_human_range:
                status = 'human_like'
                interpretation = 'Word choices feel natural and varied.'
            elif in_ai_range:
                status = 'ai_like'
                interpretation = 'Words are too predictable — a sign of AI writing.'
            elif perplexity_avg > baseline['human_range'][1]:
                status = 'human_like'
                interpretation = 'Very creative word choices — looks human.'
            else:
                status = 'uncertain'
                interpretation = 'Not enough to tell on its own.'

            fingerprint.append({
                'metric': baseline['label'],
                'value': round(perplexity_avg, 1),
                'human_range': f"{baseline['human_range'][0]}–{baseline['human_range'][1]}",
                'ai_range': f"{baseline['ai_range'][0]}–{baseline['ai_range'][1]}",
                'status': status,
                'interpretation': interpretation,
                'description': baseline['description'],
            })

        # Burstiness
        burstiness = metrics.get('burstiness', {})
        burst_score = burstiness.get('score', 0) if isinstance(burstiness, dict) else float(burstiness or 0)

        if burst_score > 0:
            baseline = self.HUMAN_BASELINES['burstiness']
            in_human = baseline['human_range'][0] <= burst_score <= baseline['human_range'][1]
            in_ai = baseline['ai_range'][0] <= burst_score <= baseline['ai_range'][1]

            if in_human:
                status = 'human_like'
                interpretation = 'Good mix of short and long sentences.'
            elif in_ai:
                status = 'ai_like'
                interpretation = 'Sentences are all similar length — typical of AI.'
            else:
                status = 'uncertain'
                interpretation = 'Could go either way — not enough to tell.'

            fingerprint.append({
                'metric': baseline['label'],
                'value': round(burst_score, 2),
                'human_range': f"{baseline['human_range'][0]}–{baseline['human_range'][1]}",
                'ai_range': f"{baseline['ai_range'][0]}–{baseline['ai_range'][1]}",
                'status': status,
                'interpretation': interpretation,
                'description': baseline['description'],
            })

        # Vocabulary richness
        vocab = metrics.get('vocabulary_richness', 0)
        if isinstance(vocab, str):
            try:
                vocab = float(vocab)
            except (ValueError, TypeError):
                vocab = 0

        if vocab > 0:
            baseline = self.HUMAN_BASELINES['vocabulary_richness']
            in_human = baseline['human_range'][0] <= vocab <= baseline['human_range'][1]
            in_ai = baseline['ai_range'][0] <= vocab <= baseline['ai_range'][1]

            if in_human:
                status = 'human_like'
                interpretation = 'Uses a wide range of different words.'
            elif in_ai:
                status = 'ai_like'
                interpretation = 'Keeps repeating the same common words.'
            elif vocab > baseline['human_range'][1]:
                status = 'human_like'
                interpretation = 'Very diverse vocabulary — looks human.'
            else:
                status = 'uncertain'
                interpretation = 'Word variety alone is not enough to tell.'

            fingerprint.append({
                'metric': baseline['label'],
                'value': round(vocab, 1),
                'human_range': f"{baseline['human_range'][0]}–{baseline['human_range'][1]}",
                'ai_range': f"{baseline['ai_range'][0]}–{baseline['ai_range'][1]}",
                'status': status,
                'interpretation': interpretation,
                'description': baseline['description'],
            })

        return fingerprint

    # ─── Per-Sentence Evidence ───────────────────────────────────────────────

    def _build_sentence_evidence(self, flagged: List, metrics: Dict, sentence_analysis: List = None) -> Dict:
        """
        Build dynamic sentence-level evidence with sentence numbers,
        actual quotes, AI scores, and pattern reasons.
        """
        all_sentences = sentence_analysis or []
        total = metrics.get('sentence_count', max(len(all_sentences), len(flagged), 1))

        if not all_sentences and not flagged:
            return {
                'total_sentences': total,
                'flagged_count': 0,
                'flagged_ratio': 0.0,
                'sentences': [],
                'summary': 'No individual sentences were analyzed.',
            }

        # Build numbered sentence list from sentence_analysis (preferred) or flagged
        sentences_data = []
        if all_sentences:
            for idx, s in enumerate(all_sentences):
                text = s.get('text', s.get('sentence', ''))
                score = s.get('ai_score', s.get('score', 0))
                is_flagged = s.get('is_flagged', score > 60)
                sent_patterns = s.get('patterns', [])

                # Build reason string from patterns
                reason = ''
                if sent_patterns:
                    pat_names = []
                    for p in sent_patterns[:3]:
                        pat_text = p.get('pattern', '')
                        pat_type = p.get('type', p.get('category', ''))
                        if pat_text:
                            pat_names.append(f'"{pat_text}"')
                        elif pat_type:
                            pat_names.append(pat_type.replace('_', ' '))
                    if pat_names:
                        reason = f'Contains: {', '.join(pat_names)}'
                elif score > 80:
                    reason = 'Very predictable structure and word choices'
                elif score > 60:
                    reason = 'Writing style matches AI patterns'

                display = text[:100] + '...' if len(text) > 100 else text

                sentences_data.append({
                    'num': idx + 1,
                    'text': display,
                    'score': round(score, 1),
                    'flagged': is_flagged,
                    'reason': reason,
                })
        else:
            # Fallback: only flagged sentences available (no numbering)
            for s in flagged:
                text = s.get('text', s.get('sentence', ''))
                score = s.get('ai_score', s.get('score', 0))
                display = text[:100] + '...' if len(text) > 100 else text
                sentences_data.append({
                    'num': 0,
                    'text': display,
                    'score': round(score, 1),
                    'flagged': True,
                    'reason': '',
                })

        flagged_count = sum(1 for s in sentences_data if s['flagged'])
        ratio = flagged_count / max(total, 1)

        # Sort all sentences by score desc, take top flagged
        flagged_items = sorted(
            [s for s in sentences_data if s['flagged']],
            key=lambda s: s['score'],
            reverse=True
        )[:5]

        # Human-like sentences (lowest scoring, for contrast)
        human_items = sorted(
            [s for s in sentences_data if not s['flagged'] and s['score'] < 40],
            key=lambda s: s['score']
        )[:2]

        if ratio > 0.6:
            summary = f'{flagged_count} of {total} sentences look AI-written — most of this text is likely generated.'
        elif ratio > 0.3:
            summary = f'{flagged_count} of {total} sentences look AI-written — the text may be partially AI-generated.'
        elif flagged_count > 0:
            summary = f'{flagged_count} of {total} sentences look AI-written — most reads as human.'
        else:
            summary = f'None of the {total} sentences were flagged as AI-written.'

        return {
            'total_sentences': total,
            'flagged_count': flagged_count,
            'flagged_ratio': round(ratio, 2),
            'sentences': flagged_items,
            'human_sentences': human_items,
            'summary': summary,
        }


if __name__ == "__main__":
    # Quick test
    explainer = TextExplainer()

    mock_result = {
        "prediction": "ai_generated",
        "confidence": 78.5,
        "scores": {"human": 21.5, "ai_generated": 78.5},
        "metrics": {
            "word_count": 150,
            "sentence_count": 8,
            "vocabulary_richness": 45.2,
            "rhythm": {"status": "Uniform", "description": "Highly consistent rhythm"},
            "burstiness": {"score": 0.2},
        },
        "detected_patterns": {
            "total_count": 5,
            "categories": {
                "formal_transitions": {"count": 3, "type": "Formal Transition", "examples": ["furthermore", "in conclusion"]},
                "hedging": {"count": 2, "type": "Hedging Language", "examples": ["it's important to note"]},
            },
        },
        "flagged_sentences": [
            {"text": "Furthermore, it is important to note that...", "ai_score": 85},
        ],
    }

    import json
    explanation = explainer.explain(mock_result)
    print(json.dumps(explanation, indent=2))
