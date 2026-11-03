import unittest
from fade_content_agent.translations import cue_translation_warnings

class CueTranslationTests(unittest.TestCase):
    def test_flags_cue_word_outside_complete_sentence_translation(self):
        cues=[{'id':'322','translation':"him don't miss it",'sentence_ids':['s1']}]
        sentences=[{'id':'s1','translation':"Don't miss it."}]
        warnings=cue_translation_warnings(cues,sentences)
        self.assertEqual(warnings[0]['reason'],'cue_translation_context_mismatch')
        self.assertEqual(warnings[0]['unmatched_tokens'],['him'])

    def test_flags_incomplete_fragment_with_no_sentence_translation_overlap(self):
        cues=[{'id':'324','translation':'an','sentence_ids':['s1']}]
        sentences=[{'id':'s1','translation':"Don't miss it."}]
        self.assertEqual(cue_translation_warnings(cues,sentences)[0]['cue_id'],'324')

    def test_accepts_contextual_cue_fragment(self):
        cues=[{'id':'1','translation':'miss it','sentence_ids':['s1']}]
        sentences=[{'id':'s1','translation':"Don't miss it."}]
        self.assertEqual(cue_translation_warnings(cues,sentences),[])

    def test_flags_empty_cue_translation(self):
        self.assertEqual(cue_translation_warnings([{'id':'1','translation':'  ','sentence_ids':[]}],[]),[
            {'reason':'empty_cue_translation','cue_id':'1'}])

