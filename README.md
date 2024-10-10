# LLM-based Business Process Documentation Generation

This approach aim to harness LLM to generate business process documentation.

## File organization

    dataset
    
    	--bpmn		# whole BPMN files
    
    	--n_rpst	# RPST corresponding to n_swd
    
    	--n_swd		# documentation corresponding to n_rpst
    
    	--swd		# documentation corresponding to tested
    
    	--tested	# RPST corresponding to swd
    
    lib		        # BLEURE metric
    
    prompt
    
    	--hct
    
    		--and	# demonstrations for <and>
    
    		--loop	# demonstrations for <loop>
    
    		...
    
    	--in_context
    
    		prompt.txt	# prompts for in-context learning
    
    result
    
    	fields.txt		# field of all BPMN
    
    	fields_96.txt		# field of 96 BPMN expect demonstrations for in-context learning
    
    	task_s.png		# statistics about number of tasks of all BPMNs
    
    src
    
    	call_llm.py		# API of call chatGPT 
    
    	evaluate.py		# evaluate the generated documentation
    
    	hierarchical_construction_technique.py		# hierarchical construction technique method
    
    	in_context_prompt	# in-context learning
    
    	metric.py		# all metrics (grammar correcness and language style, BERTScore, SBERT, BLEURT)
    
    	statistic.py		# statistic the number of any type of node