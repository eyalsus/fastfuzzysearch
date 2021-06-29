from fastfuzzysearch import FastFuzzySearch


def train(finalize=False, ngram_whitelist=None, utilize_disk=False):
    fuzzy = FastFuzzySearch(8, train_step_size=3, utilize_disk=utilize_disk)
    ssdeep_list = [
        '768:kskE8uorKordv9EWNrd2DLQL6rFXZ0Jnibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6',
        '768:kskE8uorKordv9EWNrd2DLQL6rFXZ0Jnibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6',
        '768:kskE8uoaaaaaaaaL6rFXZ0Jnibi4TKXh+2LvvzMKeqQqSbbbbbbTot:kphuorKordvtrdccccccciPyJvz75SU6',
        '24:V1a0FlxVoXIhdITYJhDp/quNg/FRSeP+fEXUXmQB8gYfXhdaVMi5Vj:na0FlxnhdIUJhDpCF3i+1JtRY/r',
        '24:V10URhx9IkjsvmCiHpCQgkg6X5AyYsXmUbQKNWgBq:nbhxgvmCiHpNg85AyuwNhBq',
        '24:VCZUZ7RPlo4akYYsUg90sV4WN8cVFsX5A97Xmhs5S8:NtViqI0fWN1D25A9qK/',
        '24:Vj+s7xVoXnpc6WdXpkUpwsph3Y4FX5MXmPS/8r3+6l:p+s7xapjWdpNpwspBt3PH3l',
        '12:Vk7LO9CdhedcY6H92V2x9Sy/lmCljUnOHCKCIp82AlIa:VsMmhRXdJScm8BHFCIaSa',
        '12:Vk7LOaaaaaaaaaaaaaax9Sy/lmCljUnObbbbbbbbbbb2AlIa:VsMmhRXdJSccccccSa',
        '12:vfvfvaaaaaaaaaaaaaaxvfvfv/vfvfvObbbbbbbbbbb2AlIa:42342Scc43243aSa',
        '12:423TESQwaaavaaaaaaaaaaaaaax423TESQwbbbbbbbbbbbbbbbIa:423TESQw',
        '12:423TESQwaaavddddddddddddddx423TESQwrrrrrrrrrrrrrrrIa:423TESQw',
    ]
    
    for idx, ssdeep_hash in enumerate(ssdeep_list):
        fuzzy.add_ssdeep_hash(ssdeep_hash, {'i': idx}, ngram_whitelist=ngram_whitelist)

    fuzzy.fit(finalize=finalize)
    return fuzzy

def test_normal_flow():
    fuzzy = train(finalize=False)
    assert len(fuzzy.lookup('24:VCZUZ7RPlo1111111111111111111111X5A97Xmhs5S8:NtViqI0fWN1D25A9qK/', one_match=True)) > 0
    assert len(fuzzy.final_automaton_ngrams) > 0

def test_finallize():
    fuzzy = train(finalize=True)
    assert len(fuzzy.lookup('24:VCZUZ7RPlo1111111111111111111111X5A97Xmhs5S8:NtViqI0fWN1D25A9qK/', one_match=True)) > 0
    assert len(fuzzy.final_automaton_ngrams) == 0

def test_not_repeating_one_match():
    fuzzy = train(finalize=True)
    assert len(fuzzy.lookup('768:kskE8uorKordv9EWNr333DLQL6rFXZ0nibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6', one_match=True)) == 1
    assert len(fuzzy.final_automaton_ngrams) == 0

def test_not_repeating_matches():
    fuzzy = train(finalize=True)
    assert len(fuzzy.lookup('768:kskE8uorKordv9EWNr333DLQL6rFXZ0nibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6', one_match=False)) == 1
    assert len(fuzzy.final_automaton_ngrams) == 0

def test_whitelist():
    # fuzzy = train(finalize=True, ngram_whitelist=set(['423TESQw', '23TESQwa', '3TESQwaa', 'TESQwaaa', 'ESQwaaav', 'SQwaaava', 'Qwaaavaa', 'waaavaaa']))
    fuzzy = train(finalize=True, ngram_whitelist=set(['423TESQw', 'dddx423T', '23TESQwa']))
    assert len(fuzzy.lookup('12:423TESQwaaavdddsssssdddddx423TESQwrrsssssssrrr2AlIa:423TESQw', one_match=False)) == 1
    fuzzy = train(finalize=True, ngram_whitelist=set(['423TESQw', 'dddx423T', '23TESQwa', 'Qwaaavdd', 'waaavddd', 'ddddx423', 'ESQwaaav', 'dddddx42', '3TESQwrr', 'dx423TES', 'SQwaaavd', 'ddx423TE', '23TESQwr', 'x423TESQ', 'TESQwaaa', '3TESQwaa']))
    assert len(fuzzy.lookup('12:423TESQwaaavdddsssssdddddx423TESQwrrsssssssrrr2AlIa:423TESQw', one_match=False)) == 0

def test_normal_flow_utilize_disk():
    fuzzy = train(finalize=False, utilize_disk=True)
    assert len(fuzzy.lookup('24:VCZUZ7RPlo1111111111111111111111X5A97Xmhs5S8:NtViqI0fWN1D25A9qK/', one_match=True)) > 0

def test_not_repeating_one_match_utilize_disk():
    fuzzy = train(finalize=True, utilize_disk=True)
    assert len(fuzzy.lookup('768:kskE8uorKordv9EWNr333DLQL6rFXZ0nibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6', one_match=True)) == 1
    assert len(fuzzy.final_automaton_ngrams) == 0

def test_not_repeating_matches_utilize_disk():
    fuzzy = train(finalize=True, utilize_disk=True)
    assert len(fuzzy.lookup('768:kskE8uorKordv9EWNr333DLQL6rFXZ0nibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6', one_match=False)) == 1
    assert len(fuzzy.final_automaton_ngrams) == 0

def test_whitelist_utilize_disk():
    # fuzzy = train(finalize=True, ngram_whitelist=set(['423TESQw', '23TESQwa', '3TESQwaa', 'TESQwaaa', 'ESQwaaav', 'SQwaaava', 'Qwaaavaa', 'waaavaaa']))
    fuzzy = train(finalize=True, ngram_whitelist=set(['423TESQw', 'dddx423T', '23TESQwa']), utilize_disk=True)
    assert len(fuzzy.lookup('12:423TESQwaaavdddsssssdddddx423TESQwrrsssssssrrr2AlIa:423TESQw', one_match=False)) == 1
    fuzzy = train(finalize=True, ngram_whitelist=set(['423TESQw', 'dddx423T', '23TESQwa', 'Qwaaavdd', 'waaavddd', 'ddddx423', 'ESQwaaav', 'dddddx42', '3TESQwrr', 'dx423TES', 'SQwaaavd', 'ddx423TE', '23TESQwr', 'x423TESQ', 'TESQwaaa', '3TESQwaa']))
    assert len(fuzzy.lookup('12:423TESQwaaavdddsssssdddddx423TESQwrrsssssssrrr2AlIa:423TESQw', one_match=False)) == 0

