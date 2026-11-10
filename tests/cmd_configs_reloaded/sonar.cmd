get attributes examples/sonar_example/sonar.attr
get ontofilename libs/sonar_ontology.owl
get ontology examples/sonar_example/sonar.onto
get training_examples examples/sonar_example/sonar.train.pat
get test_examples examples/sonar_example/sonar.test.pat
get network examples/sonar_example/sonar
set use_ontology 1
set seed 10
set tree_size_limit 10
set min_sample 5000
lo_mofn examples/sonar_example/sonar.fidelity
test_fidelity
test_correctness
print_tree
draw_tree_revisited examples/sonar_example/sonar.dot
print_rules examples/sonar_example/sonar.rules
quit
