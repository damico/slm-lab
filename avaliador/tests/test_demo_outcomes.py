"""The reference continuation must stay separate from the prefix used by the SLM."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'lab04-decisoes'))
from demo_outcomes import build_outcomes, history_fingerprint


def corpus():
    original = [['customer', 'When does my code expire?'],
                ['action', 'System Action: search timing'],
                ['agent', 'Promo codes expire after seven days.'],
                ['customer', 'Thank you.']]
    targets = [[None, None, None], ['timing', 'take_action', 'search-timing'],
               ['timing', 'retrieve_utterance', None], [None, None, None]]
    return {'test': [{'convo_id': 4989, 'original': original,
                     'delexed': [{'targets': t} for t in targets]}]}


class OutcomesTest(unittest.TestCase):
    def test_preserves_actual_reply_actions_and_final_event(self):
        outcome = build_outcomes(corpus(), [{'id': '4989:1', 'expected': 'search-timing'}])['4989:1']
        self.assertEqual(outcome['target']['action'], 'search-timing')
        self.assertEqual(outcome['delivered_messages'][0]['text'], 'Promo codes expire after seven days.')
        self.assertEqual(outcome['last_action']['id'], '4989:1')
        self.assertEqual(outcome['final_event']['role'], 'customer')
        self.assertEqual(len(outcome['continuation']), 3)
        self.assertEqual(outcome['history_sha256'], history_fingerprint([
            {'role': 'customer', 'text': 'When does my code expire?'}]))

    def test_refuses_reference_label_mismatch(self):
        with self.assertRaises(ValueError):
            build_outcomes(corpus(), [{'id': '4989:1', 'expected': 'offer-refund'}])

    def test_previous_actions_are_part_of_the_prefix_fingerprint(self):
        outcome = build_outcomes(corpus(), [{'id': '4989:2', 'expected': 'retrieve_utterance'}])['4989:2']
        self.assertEqual(outcome['history_sha256'], history_fingerprint([
            {'role': 'customer', 'text': 'When does my code expire?'},
            {'role': 'action', 'text': 'System Action: search timing', 'action': 'search-timing'}]))


if __name__ == '__main__':
    unittest.main()
