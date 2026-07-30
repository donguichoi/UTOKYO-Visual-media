**Use of generative AI **

Generative AI tools were used as assistants in this assignment. All final decisions on experimental design, code execution, and interpretation of results were made by me.

**Tools used**

• Perplexity: initial Google Colab environment setup

• Claude: code debugging (majority), diagnostic logging, English language grammar polishing

**Main purposes**

• Organising the initial installation order, MANO file placement, and execution commands for reproducing the official HaMeR implementation on Google Colab

• Debugging code errors and verifying output files during execution

• Interpreting the pipeline structure of the official demo.py and assisting with the diagnostic logging code

• Refining the wording and sentence structure of the author's English draft for academic register.

**Parts revised or judged by the author**

• I found that one suggested patch was placed after an early-exit branch, where it would never execute in exactly the failure cases it was meant to handle, and relocated it after tracing the control flow.

• From the diagnostic log, I observed that the two candidates in palm_closeup_1 were separated by only 0.005 in mean keypoint confidence and independently derived Improvement 2.

• I independently verified reproducibility by repeating the full experiment three or more times and confirming that all reported results reflect direct execution.

• The failure analysis, the interpretation of the improvement, and all conclusions were written by myself.
