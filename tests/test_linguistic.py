import unittest
from fade_content_agent.linguistic import normalize_concept, merge_concept, map_occurrences, exclude_zero_occurrence_concepts, normalize_occurrence_mappings, limit_word_occurrences, annotated_sentence_ids, limit_annotated_sentence_coverage
from fade_content_agent.subtitles import Cue

class LinguisticTests(unittest.TestCase):
    def concept(self, lemma, category, meaning, difficulty=1, selection_priority='medium'):
        result,warning=normalize_concept({'lemma':lemma,'category':category,'meaning_key':meaning,'difficulty':difficulty,'translation':'meaning','explanation':'learning note','selection_priority':selection_priority,'selection_reason':'useful learning item'})
        self.assertIsNone(warning); return result
    def test_basic_vocabulary_is_valid(self):
        concept=self.concept('ser','verb','to be: identity or characteristic',1)
        self.assertEqual(concept['difficulty'],1); self.assertEqual(concept['frequency'],0)
        cue=Cue('1',1,0,1,'Yo soy estudiante.')
        occurrences,_,frequency,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['soy'],'sentence_ids':['s1']}],[cue],[{'id':'s1','cue_ids':['1']}])
        self.assertEqual(occurrences[0]['text'],'soy'); self.assertEqual(frequency[concept['concept_id']],1)
    def test_fixed_expression_and_idiom_categories_are_supported(self):
        fixed=self.concept('darse cuenta','fixed_expression','realize',2)
        idiom=self.concept('estar en las nubes','idiom','be distracted',3)
        self.assertNotEqual(fixed['concept_id'],idiom['concept_id'])
    def test_repeated_words_and_inflected_forms_share_one_concept(self):
        concept=self.concept('hablar','verb','to speak')
        cues=[Cue('1',1,0,2,'Yo hablo y tú hablas.'),Cue('2',2,2,4,'Ayer hablamos.')]
        sentences=[{'id':'s1','cue_ids':['1']},{'id':'s2','cue_ids':['2']}]
        mappings=[{'concept_id':concept['concept_id'],'surface_forms':['hablo','hablas'],'sentence_ids':['s1']},{'concept_id':concept['concept_id'],'surface_forms':['hablamos'],'sentence_ids':['s2']}]
        occ,warnings,frequency,sentence_concepts=map_occurrences({concept['concept_id']:concept},mappings,cues,sentences)
        self.assertEqual(len(occ),3); self.assertEqual(frequency[concept['concept_id']],3); self.assertFalse(warnings)
        self.assertEqual(sentence_concepts['s1'],[concept['concept_id']])
    def test_distinct_senses_of_same_word_get_distinct_concepts(self):
        money=self.concept('banco','noun','financial institution')
        seat=self.concept('banco','noun','bench for sitting')
        cues=[Cue('1',1,0,1,'Voy al banco.'),Cue('2',2,1,2,'Me siento en el banco.')]
        sentences=[{'id':'s1','cue_ids':['1']},{'id':'s2','cue_ids':['2']}]
        mappings=[{'concept_id':money['concept_id'],'surface_forms':['banco'],'sentence_ids':['s1']},{'concept_id':seat['concept_id'],'surface_forms':['banco'],'sentence_ids':['s2']}]
        occ,_,_,_=map_occurrences({money['concept_id']:money,seat['concept_id']:seat},mappings,cues,sentences)
        self.assertEqual({o['concept_id'] for o in occ},{money['concept_id'],seat['concept_id']})
    def test_cross_cue_expression_creates_linked_local_occurrences(self):
        concept=self.concept('por qué','fixed_expression','why')
        cues=[Cue('1',1,0,1,'¿Por'),Cue('2',2,1,2,'qué vienes?')]
        sentences=[{'id':'s1','cue_ids':['1','2']}]
        mapping=[{'concept_id':concept['concept_id'],'surface_forms':['Por qué'],'sentence_ids':['s1']}]
        occurrences,warnings,frequency,_=map_occurrences({concept['concept_id']:concept},mapping,cues,sentences)
        self.assertEqual([x['text'] for x in occurrences],['Por','qué'])
        self.assertEqual(len({x['instance_id'] for x in occurrences}),1)
        self.assertEqual(frequency[concept['concept_id']],1); self.assertFalse(warnings)
        self.assertEqual([(x['start_char'],x['end_char']) for x in occurrences],[(1,4),(0,3)])
    def test_expression_crosses_non_linguistic_interruption_without_annotating_it(self):
        concept=self.concept('no te lo pierdas','fixed_expression','do not miss it',2)
        cues=[Cue('322',322,1,2,'él no te lo'),Cue('323',323,2,3,'[Aplausos]'),Cue('324',324,3,4,'pierdas an')]
        sentences=[{'id':'s1','cue_ids':['322','324']}]
        mappings=[{'concept_id':concept['concept_id'],'surface_forms':['no te lo pierdas'],'sentence_ids':['s1']}]
        occurrences,warnings,frequency,_=map_occurrences({concept['concept_id']:concept},mappings,cues,sentences)
        self.assertEqual([(o['cue_id'],o['text']) for o in occurrences],[('322','no te lo'),('324','pierdas')])
        self.assertEqual([(o['start_char'],o['end_char']) for o in occurrences],[(3,11),(0,7)])
        self.assertEqual(len({o['instance_id'] for o in occurrences}),1)
        self.assertEqual(frequency[concept['concept_id']],1)
        self.assertFalse(any(o['cue_id']=='323' for o in occurrences))
        self.assertFalse(warnings)
    def test_decomposed_unicode_matches_and_preserves_codepoint_offsets(self):
        concept=self.concept('sentarse','verb','to sit down')
        cue=Cue('1',1,0,1,'Sie\u0301ntate')
        sentence={'id':'s1','cue_ids':['1']}
        occurrences,_,_,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['Siéntate'],'sentence_ids':['s1']}],[cue],[sentence])
        self.assertEqual(occurrences[0]['text'],cue.text)
        self.assertEqual(cue.text[occurrences[0]['start_char']:occurrences[0]['end_char']],occurrences[0]['text'])
        self.assertEqual(occurrences[0]['end_char'],len(cue.text))
    def test_hyphen_and_space_expression_variants_match(self):
        concept=self.concept('por qué','fixed_expression','why')
        cue=Cue('1',1,0,1,'por-qué')
        sentence={'id':'s1','cue_ids':['1']}
        occurrences,_,_,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['por qué'],'sentence_ids':['s1']}],[cue],[sentence])
        self.assertEqual(occurrences[0]['text'],'por-qué')
    def test_noun_plural_variant_maps_to_exact_surface(self):
        concept=self.concept('corresponsal','noun','pen pal',3)
        cue=Cue('1',1,0,1,'Somos corresponsales.')
        sentence={'id':'s1','cue_ids':['1']}
        occurrences,warnings,frequency,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['corresponsal'],'sentence_ids':['s1']}],[cue],[sentence],source_language='es')
        self.assertEqual([o['text'] for o in occurrences],['corresponsales'])
        self.assertEqual(frequency[concept['concept_id']],1)
        self.assertFalse(any(w['reason']=='unmatched_occurrence' for w in warnings))
    def test_spanish_morphology_is_not_applied_to_other_source_languages(self):
        concept=self.concept('corresponsal','noun','pen pal',3)
        cue=Cue('1',1,0,1,'corresponsales')
        sentence={'id':'s1','cue_ids':['1']}
        occurrences,warnings,_,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['corresponsal'],'sentence_ids':['s1']}],[cue],[sentence],source_language='en')
        self.assertEqual(occurrences,[])
        self.assertTrue(any(w['reason']=='unmatched_occurrence' for w in warnings))
    def test_french_elided_form_is_located_with_exact_unicode_offsets(self):
        concept=self.concept('aimer','verb','to like',2)
        cue=Cue('1',1,0,1,"J'aime beaucoup l'été.")
        sentence={'id':'s1','cue_ids':['1']}
        occurrences,warnings,frequency,_=map_occurrences(
            {concept['concept_id']:concept},
            [{'concept_id':concept['concept_id'],'surface_forms':["aime"],'sentence_ids':['s1']}],
            [cue],[sentence],source_language='fr-FR')
        start=cue.text.index('aime')
        self.assertEqual(len(occurrences),1)
        self.assertEqual((occurrences[0]['start_char'],occurrences[0]['end_char']),(start,start+len('aime')))
        self.assertEqual(cue.text[occurrences[0]['start_char']:occurrences[0]['end_char']],'aime')
        self.assertEqual(frequency[concept['concept_id']],1)
        self.assertFalse(warnings)
    def test_french_accented_surface_keeps_exact_text_and_offsets(self):
        concept=self.concept('été','noun','summer',1)
        cue=Cue('1',1,0,1,"J'aime l'été.")
        occurrences,warnings,_,_=map_occurrences(
            {concept['concept_id']:concept},
            [{'concept_id':concept['concept_id'],'surface_forms':['été'],'sentence_ids':['s1']}],
            [cue],[{'id':'s1','cue_ids':['1']}],source_language='fr')
        self.assertEqual(len(occurrences),1)
        self.assertEqual(occurrences[0]['text'],'été')
        self.assertEqual(cue.text[occurrences[0]['start_char']:occurrences[0]['end_char']],'été')
        self.assertFalse(warnings)
    def test_french_expression_can_cross_adjacent_cues(self):
        concept=self.concept('avoir besoin de','fixed_expression','to need',2)
        cues=[Cue('1',1,0,1,'J’ai besoin'),Cue('2',2,1,2,'de repos.')]
        sentence={'id':'s1','cue_ids':['1','2']}
        occurrences,warnings,frequency,_=map_occurrences(
            {concept['concept_id']:concept},
            [{'concept_id':concept['concept_id'],'surface_forms':['besoin de'],'sentence_ids':['s1']}],
            cues,[sentence],source_language='fr')
        self.assertEqual([item['text'] for item in occurrences],['besoin','de'])
        self.assertEqual(len({item['instance_id'] for item in occurrences}),1)
        self.assertEqual(frequency[concept['concept_id']],1)
        self.assertFalse(warnings)
    def test_alternative_form_mismatch_is_distinct_from_missing_context(self):
        concept=self.concept('acordarse','verb','to remember',2)
        cue=Cue('1',1,0,1,'¿Te acuerdas?')
        sentence={'id':'s1','cue_ids':['1']}
        _,warnings,_,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['me acuerdo','te acuerdas'],'sentence_ids':['s1']}],[cue],[sentence])
        reasons={w['reason'] for w in warnings}
        self.assertIn('alternative_surface_mismatch',reasons)
        self.assertNotIn('unmatched_occurrence',reasons)
    def test_overlapping_distinct_concepts_are_retained_and_warned(self):
        verb=self.concept('dar','verb','to give')
        expression=self.concept('dar cuenta','fixed_expression','to realize')
        cues=[Cue('1',1,0,1,'dar cuenta de algo')]; sentences=[{'id':'s1','cue_ids':['1']}]
        maps=[{'concept_id':verb['concept_id'],'surface_forms':['dar'],'sentence_ids':['s1']},{'concept_id':expression['concept_id'],'surface_forms':['dar cuenta'],'sentence_ids':['s1']}]
        occurrences,warnings,_,_=map_occurrences({verb['concept_id']:verb,expression['concept_id']:expression},maps,cues,sentences)
        self.assertEqual(len(occurrences),2)
        self.assertEqual({o['concept_id'] for o in occurrences},{verb['concept_id'],expression['concept_id']})
    def test_overlapping_surface_forms_prefer_canonical_lemma_and_warn(self):
        concept=self.concept('factura','noun','bill')
        cue=Cue('1',1,0,1,'la factura del teléfono')
        sentence={'id':'s1','cue_ids':['1']}
        mapping=[{'concept_id':concept['concept_id'],'surface_forms':['factura del','factura'],'sentence_ids':['s1']}]
        occurrences,warnings,_,_=map_occurrences({concept['concept_id']:concept},mapping,[cue],[sentence])
        self.assertEqual([item['text'] for item in occurrences],['factura'])
        self.assertTrue(any(item['reason']=='redundant_overlap_same_concept' for item in warnings))
    def test_same_surface_distinct_concepts_is_recorded_as_ambiguous(self):
        one=self.concept('banco','noun','financial institution')
        two=self.concept('banco','noun','bench')
        cues=[Cue('1',1,0,1,'banco')]; sentences=[{'id':'s1','cue_ids':['1']}]
        maps=[{'concept_id':one['concept_id'],'surface_forms':['banco'],'sentence_ids':['s1']},{'concept_id':two['concept_id'],'surface_forms':['banco'],'sentence_ids':['s1']}]
        _,warnings,_,_=map_occurrences({one['concept_id']:one,two['concept_id']:two},maps,cues,sentences)
        self.assertTrue(any(w['reason']=='same_span_multiple_concepts' for w in warnings))
    def test_unmatched_mapping_is_a_warning(self):
        concept=self.concept('mesa','noun','table')
        cues=[Cue('1',1,0,1,'silla')]; sentences=[{'id':'s1','cue_ids':['1']}]
        _,warnings,frequency,_=map_occurrences({concept['concept_id']:concept},[{'concept_id':concept['concept_id'],'surface_forms':['mesa'],'sentence_ids':['s1']}],cues,sentences)
        self.assertEqual(frequency,{}); self.assertTrue(any(w['reason']=='unmatched_occurrence' for w in warnings))
    def test_concept_ids_are_stable_for_canonical_forms(self):
        a=self.concept('hablar','verb','to speak')
        b=self.concept('  HABLAR  ','verb','to speak')
        self.assertEqual(a['concept_id'],b['concept_id'])
    def test_same_concept_from_multiple_batches_keeps_one_definition(self):
        catalog={}
        first={'lemma':'hablar','category':'verb','meaning_key':'to speak','difficulty':1,'translation':'speak','explanation':'canonical note'}
        repeated={**first,'lemma':' HABLAR ','explanation':'slightly different wording'}
        cid,warning=merge_concept(catalog,first)
        second_id,warning2=merge_concept(catalog,repeated)
        self.assertEqual(cid,second_id); self.assertEqual(len(catalog),1)
        self.assertEqual(catalog[cid]['explanation'],'canonical note')
        self.assertEqual(warning2['reason'],'concept_metadata_conflict')
    def test_zero_occurrence_concepts_are_excluded_with_separate_warning(self):
        present=self.concept('hola','noun','greeting')
        absent=self.concept('mesa','noun','table')
        present.update(frequency=1,occurrence_ids=['lo1'])
        absent.update(frequency=0,occurrence_ids=[])
        published,warnings=exclude_zero_occurrence_concepts({present['concept_id']:present,absent['concept_id']:absent})
        self.assertEqual(set(published),{present['concept_id']})
        self.assertEqual(warnings,[{'reason':'zero_occurrence_concept_not_published','concept_id':absent['concept_id'],'lemma':'mesa','category':'noun'}])
    def test_occurrence_response_requires_one_exact_sentence_context(self):
        concept=self.concept('hablar','verb','to speak')
        normalized,warnings=normalize_occurrence_mappings([
            {'concept_id':concept['concept_id'],'sentence_id':'s1','surface_forms':['hablo','hablas']},
            {'concept_id':concept['concept_id'],'sentence_id':'s2','surface_forms':[]},
            {'concept_id':concept['concept_id'],'sentence_id':'s9','surface_forms':['habla']}],
            {concept['concept_id']},{'s1','s2'})
        self.assertEqual(normalized,[{'concept_id':concept['concept_id'],'surface_forms':['hablo','hablas'],'sentence_ids':['s1']}])
        self.assertEqual([w['reason'] for w in warnings],['incomplete_occurrence_mapping','out_of_batch_sentence_reference'])

    def test_word_occurrence_cap_prefers_priority_and_warns(self):
        high=self.concept('casa','noun','house',1,'high')
        medium=self.concept('correr','verb','run',1,'medium')
        low=self.concept('grande','adjective','big',1,'low')
        concepts={c['concept_id']:c for c in (high,medium,low)}
        cue=Cue('1',1,0,1,'casa corre grande')
        sentence={'id':'s1','cue_ids':['1']}
        mappings=[
            {'concept_id':high['concept_id'],'surface_forms':['casa'],'sentence_ids':['s1']},
            {'concept_id':medium['concept_id'],'surface_forms':['corre'],'sentence_ids':['s1']},
            {'concept_id':low['concept_id'],'surface_forms':['grande'],'sentence_ids':['s1']},
        ]
        occurrences,_,_,_=map_occurrences(concepts,mappings,[cue],[sentence],source_language='es')
        kept,warnings=limit_word_occurrences(occurrences,concepts,[sentence])
        self.assertEqual({item['concept_id'] for item in kept},{high['concept_id'],medium['concept_id']})
        self.assertEqual([item['reason'] for item in warnings],['single_word_sentence_cap'])

    def test_word_occurrence_cap_keeps_expression_occurrences(self):
        word=self.concept('dar','verb','give',1,'high')
        other=self.concept('casa','noun','house',1,'medium')
        expression=self.concept('dar cuenta','fixed_expression','realize',2,'high')
        concepts={c['concept_id']:c for c in (word,other,expression)}
        cue=Cue('1',1,0,1,'dar cuenta casa')
        sentence={'id':'s1','cue_ids':['1']}
        mappings=[
            {'concept_id':word['concept_id'],'surface_forms':['dar'],'sentence_ids':['s1']},
            {'concept_id':other['concept_id'],'surface_forms':['casa'],'sentence_ids':['s1']},
            {'concept_id':expression['concept_id'],'surface_forms':['dar cuenta'],'sentence_ids':['s1']},
        ]
        occurrences,_,_,_=map_occurrences(concepts,mappings,[cue],[sentence],source_language='es')
        kept,_=limit_word_occurrences(occurrences,concepts,[sentence],max_per_sentence=1)
        self.assertIn(expression['concept_id'],{item['concept_id'] for item in kept})
        self.assertEqual(sum(item['concept_id'] in {word['concept_id'],other['concept_id']} for item in kept),1)

    def test_coverage_cap_uses_semantic_sentences_and_prioritizes_candidates(self):
        high=self.concept('útil','adjective','useful',2,'high')
        low=self.concept('cosa','noun','thing',1,'low')
        concepts={high['concept_id']:high,low['concept_id']:low}
        sentences=[{'id':f's{i}'} for i in range(1,6)]
        occurrences=[
            {'occurrence_id':'o1','concept_id':low['concept_id'],'sentence_ids':['s1']},
            {'occurrence_id':'o2','concept_id':high['concept_id'],'sentence_ids':['s2']},
            {'occurrence_id':'o3','concept_id':low['concept_id'],'sentence_ids':['s3']},
            {'occurrence_id':'o4','concept_id':low['concept_id'],'sentence_ids':['s4']},
        ]
        kept,warnings,stats=limit_annotated_sentence_coverage(occurrences,concepts,sentences)
        self.assertEqual(annotated_sentence_ids(kept,sentences),{'s1','s2','s3'})
        self.assertEqual(stats,{'sentence_count':5,'annotated_sentence_count':3,'max_annotated_sentence_count':3})
        self.assertEqual([warning['sentence_id'] for warning in warnings],['s4'])

    def test_coverage_cap_prunes_dropped_sentence_reference_from_shared_occurrence(self):
        concept=self.concept('hola','noun','greeting',1,'high')
        occurrence={'occurrence_id':'o1','concept_id':concept['concept_id'],'sentence_ids':['s1','s2','s3']}
        kept,warnings,stats=limit_annotated_sentence_coverage([occurrence],{concept['concept_id']:concept},[{'id':'s1'},{'id':'s2'},{'id':'s3'}])
        self.assertEqual(kept[0]['sentence_ids'],['s1'])
        self.assertEqual(annotated_sentence_ids(kept,[{'id':'s1'},{'id':'s2'},{'id':'s3'}]),{'s1'})
        self.assertEqual(len(warnings),2)
        self.assertEqual(stats['max_annotated_sentence_count'],1)
