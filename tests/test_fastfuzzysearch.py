from fastfuzzysearch import FastFuzzySearch

def test_normal_flow():
    fuzzy = FastFuzzySearch(10, train_step_size=3)
    ssdeep_list = [
        '768:kskE8uorKordv9EWNrd2DLQL6rFXZ0Jnibi4TKXh+2LvvzMKeqQqSnUSPNanTot:kphuorKordvtrd2nrFXmhiPyJvz75SU6',
        '24:V1a0FlxVoXIhdITYJhDp/quNg/FRSeP+fEXUXmQB8gYfXhdaVMi5Vj:na0FlxnhdIUJhDpCF3i+1JtRY/r',
        '24:V10URhx9IkjsvmCiHpCQgkg6X5AyYsXmUbQKNWgBq:nbhxgvmCiHpNg85AyuwNhBq',
        '24:VCZUZ7RPlo4akYYsUg90sV4WN8cVFsX5A97Xmhs5S8:NtViqI0fWN1D25A9qK/',
        '24:Vj+s7xVoXnpc6WdXpkUpwsph3Y4FX5MXmPS/8r3+6l:p+s7xapjWdpNpwspBt3PH3l',
        '12:Vk7LO9CdhedcY6H92V2x9Sy/lmCljUnOHCKCIp82AlIa:VsMmhRXdJScm8BHFCIaSa'
    ]
    
    for idx, ssdeep_hash in enumerate(ssdeep_list):
        fuzzy.add_ssdeep_hash(ssdeep_hash, {'i': idx, 'ssdeep': ssdeep_hash})

    fuzzy.fit()

    fuzzy.lookup('24:VCZUZ7RPlo1111111111111111111111X5A97Xmhs5S8:NtViqI0fWN1D25A9qK/', one_match=True)