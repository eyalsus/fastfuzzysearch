
import re
import pyssdeep
from fastsearch import FastSearch 

COLON_PATTERN = re.compile(':')

class FastFuzzySearch:
    def __init__(self, ngram_length, train_step_size=1000):
        self.ngram_length = ngram_length
        self.train_step_size = train_step_size
        self.fastsearch = FastSearch(COLON_PATTERN, self.ngram_length)
        self.is_trained = False
        self.kb_size = 0
        self.final_automaton_ngrams = {}

    def add_ssdeep_hash(self, ssdeep_hash, descriptor):
        ngram_set = self.fastsearch.add_sentence(ssdeep_hash, selection_start=1, append_automaton=False)

        if self.kb_size > 0 and self.kb_size % self.train_step_size == 0:
            self.fit()
        
        descriptor['ssdeep'] = ssdeep_hash
        matches = None
        if self.is_trained:
            matches = self.lookup(ssdeep_hash, one_match=True)
    
        if not matches:
            for ngram in ngram_set:
                if ngram not in self.final_automaton_ngrams:
                    self.final_automaton_ngrams[ngram] = {'appearances': 1, 'descriptor': descriptor}
                else:
                    self.final_automaton_ngrams[ngram]['appearances'] += 1
            self.kb_size += 1
        else:
            for ngram in ngram_set:
                if ngram in self.final_automaton_ngrams:
                    self.final_automaton_ngrams[ngram]['appearances'] += 1

    
    def fit(self, finalize=False):
        self.fastsearch = FastSearch(COLON_PATTERN, self.ngram_length)
        for word, descriptor in self.final_automaton_ngrams.items():
            if descriptor['appearances'] > 1 or self.kb_size < 10000:
                self.fastsearch.add_sentence(word, descriptor=descriptor['descriptor'])
        self.fastsearch.fit()
        if finalize:
            self.final_automaton_ngrams.clear()
        self.is_trained = True


    def lookup(self, ssdeep_hash, one_match=False, similarity_threshold=50):
        results = []
        matches = self.fastsearch.lookup(ssdeep_hash, one_match=one_match)
        for match in matches:
            score = pyssdeep.compare(ssdeep_hash, match['ssdeep'])
            if score > similarity_threshold:
                match['score'] = score
                results.append(match)
        return results
