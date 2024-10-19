# 데이터의 구조가 이렇게 생겼다 정도만 파악해도 될듯?
# architecture가 주 contribution인 논문이라 experiments는 다음 두 데이터셋만 가지고 진행했음, 비슷한듯 다른 두 데이터임

benchmarks = {
    'MIntRec': {
        'intent_labels': [
            'Complain', 'Praise', 'Apologies', 'Thank', 'Criticize',
            'Care', 'Agree', 'Taunt', 'Flaunt',
            'Joke', 'Oppose',
            'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave',
            'Prevent', 'Greet', 'Ask for help'
        ],
        'binary_maps': {
                    'Complain': 'Emotion', 'Praise': 'Emotion', 'APologies': 'Emotion', 'Thank': 'Emotion', 'Criticize': 'Emotion',
                    'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
                    'JokePraise': 'Emotion', 'Praise': 'Emotion',
                    'Inform': 'Goal', 'Advise': 'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave': 'Goal',
                    'Prevent': 'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal' 

        }, 
        'label_len': 4, 
        'binary_intent_labels': ['Emotion', 'Goal'],
        'max_seq_length': {
            'text': 30,
            'audio_feats': 480,
            'video_feats': 230
        },
        'feat_dims':{
            'text':768,
            'audio': 768,
            'video': 1024
        }
    },

    'MELD':{
        'intent_labels':[
                    'Greeting', 'Question', 'Answer', 'Statement Opinion', 'Statement Non Opinion',
                    'Apology', 'Command', 'Agreement', 'Disagreement',
                    'Ackonwledge', 'Backchannel', 'Others'
        ],
        'label_maps':{
                    'g': 'Greeting', 'q':'Question', 'ans':'Answer', 'o':'Statement Opinion', 's':'Statement Non Opinion',
                    'ap':'Apology', 'c':'Command', 'ag':'Agreement', 'dag':'Disagreement',
                    'a':'Ackonwledge', 'b':'Backchannel', 'oth':'Others'
        },
        'label_len':3,
        'max_seq_lengths':{
            'text': 70,
            'audio_feats': 530, 
            'video_feats': 250
        },
        'feat_dims':{
            'text': 768,
            'audio': 768,
            'video': 1024
        }
    }
}