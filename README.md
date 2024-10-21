The original code is here: https://github.com/thuiar/TCL-MAP <br>
SO Worthy to read! <br>
They merged multimodal inputs from the very upper, token level. That's why I'm dissecting every code line by line <br>

</pre>
TCL_MAP / <br>
│ <br>
├── data/ <br>
│ ├── MMDataset.py <br>
│ ├── __init__.py <br>
│ ├── base.py <br>
│ ├── mm_pre.py <br>
│ ├── text_pre.py <br>
│ └── utils.py <br>
│ <br>
├── methods <br>
│  ├── TCL_MAP <br>
│    ├── SubNets <br>
│       ├── transformers_encoder <br>
│         ├── __init__.py <br>
│         ├── multihead_attention.py <br>
│         ├── position_embeddings.py <br>
│         ├── transformers.py <br>
│       ├── FeatureNets.py <br>
│    │── AlignNets.py <br>
│    │── loss.py <br>
│    │── manager.py <br>
│    │── model.py <br>
│ <br>
├── utils <br>
│ │── tokenizer.py <br>
│ │── functions.py <br>
│ ├── metrics.py <br>
│ <br>
└── README.md <be>



</pre>
TCL_MAP / <br>
│ <br>
├── data/ <br>
│   ├── MMDataset.py <br>  
│   ├── __init__.py <br>
│   ├── base.py <br>
│   ├── mm_pre.py <br>
│   ├── text_pre.py <br>
│   └── utils.py   <br>
│  
├── methods <br> 
│   ├── TCL_MAP <br> 
│       ├── SubNets <br>
│           ├── transformers_encoder  <br> 
│               ├── __init__.py <br> 
│               ├── multihead_attention.py <br>
│               ├── position_embeddings.py <br> 
│               ├── transformers.py <br>
│           ├── FeatureNets.py <br>
│       ├── AlignNets.py  <br>
│       ├── loss.py <br> 
│       ├── manager.py <br>  
│       ├── model.py  <br>
│ <br>  
├── utils <br>
│   ├── tokenizer.py <br>  
│   ├── functions.py <br>
│   ├── metrics.py <br> 
│ <br>
└── README.md <br>
</pre>
