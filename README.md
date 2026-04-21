# Master Thesis Repository about Self-Organization of MAS

- gpt-4o
- claude-sonnet-4
- anthropic--claude-4.5-sonnet
- the gemini one
- amazon--nova-pro
- mistralai--mistral-large-instruct


### Preliminary findings

**Period-2 limit cycle**:
The run [results/interesting/20260414_114850_mistral-large_N2_T10_q56.json](results/interesting/20260414_114850_mistral-large_N2_T10_q56.json) shows a clean period-2 limit cycle under "no self-memory + homogeneous models + N=2" is a genuine motif observation and a useful
negative control: it shows that without self-memory, the system collapses into trivial imitation dynamics regardless of the actual reasoning. That's a real data point for the memory-architecture axis.

**Unwilligness to explore other options**: Usually the LLM-based MAS votes for the option that has the most voters at t=0. This is inherently flawed. Furthermore, the agents are unable to explore new options but just switch options that are represented at t=0. 