import os,sys,json
import yaml
from ipywidgets import widgets
from IPython.display import display, clear_output
from tqdm import tqdm

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname('PsycoreTestRunner.py'), ".."))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import PsycoreTestRunner
from src.results import ResultManager
import asyncio


class VariationType:
    def __init__(self, bert_graph: bool, llm_graph: bool, aws_embedding: bool, api_limited: bool, hardware_limited: bool, config_path: str):
        self.bert_graph = bert_graph
        self.llm_graph = llm_graph
        self.aws_embedding = aws_embedding
        self.api_limited = api_limited
        self.hardware_limited = hardware_limited
        self.config_path = config_path

    def __str__(self):
        return f"VariationType(bert_graph={self.bert_graph}, llm_graph={self.llm_graph}, aws_embedding={self.aws_embedding}, api_limited={self.api_limited}, hardware_limited={self.hardware_limited}, config_path={self.config_path})"
    
    @staticmethod
    def split_config(variations: list['VariationType']):
        config_structure = {
            "General": {
                "llm_graph" :{
                    "aws_embedding" :[],
                    "clip_embedding": []
                },
                "bert_graph": {
                    "aws_embedding" :[],
                    "clip_embedding": []
                }
            },
            "API_LIMITED": {
                "llm_graph" :{
                    "aws_embedding" :[],
                    "clip_embedding": []
                },
                "bert_graph": {
                    "aws_embedding" :[],
                    "clip_embedding": []
                }
            },
            "HARDWARE_LIMITED": {
                "llm_graph" :{
                    "aws_embedding" :[],
                    "clip_embedding": []
                },
                "bert_graph": {
                    "aws_embedding" :[],
                    "clip_embedding": []
                }
            },
            "API_HARDWARE_LIMITED": {
                "llm_graph" :{
                    "aws_embedding" :[],
                    "clip_embedding": []
                },
                "bert_graph": {
                    "aws_embedding" :[],
                    "clip_embedding": []
                }
            }
        }
        print("Starting split_config with variations:", [str(v) for v in variations])
        
        for variation in variations:
            # Determine which category to use
            category = "General"
            if variation.api_limited and variation.hardware_limited:
                category = "API_HARDWARE_LIMITED"
            elif variation.api_limited:
                category = "API_LIMITED"
            elif variation.hardware_limited:
                category = "HARDWARE_LIMITED"
            
            print(f"\nProcessing variation: {variation}")
            print(f"Selected category: {category}")
            
            # Determine which graph type to use
            graph_type = "llm_graph" if variation.llm_graph else "bert_graph"
            print(f"Selected graph_type: {graph_type}")
            
            # Determine which embedding type to use
            embedding_type = "aws_embedding" if variation.aws_embedding else "clip_embedding"
            print(f"Selected embedding_type: {embedding_type}")
            
            # Add the config path to the appropriate list
            config_structure[category][graph_type][embedding_type].append(variation.config_path)
            print(f"Added config path {variation.config_path} to {category}.{graph_type}.{embedding_type}")
            print(f"Current state of that list: {config_structure[category][graph_type][embedding_type]}")

        print("\nFinal config_structure:", json.dumps(config_structure, indent=2))
        return config_structure
    
    @staticmethod
    def group_by_preprocessing(config_structure: dict):
        preprocessing_groups = {
            'llm_graph_aws_embedding': [],
            'llm_graph_clip_embedding': [],
            'bert_graph_aws_embedding': [],
            'bert_graph_clip_embedding': []
        }
        
        print("Starting group_by_preprocessing with config_structure:", json.dumps(config_structure, indent=2))
        
        for category, category_data in config_structure.items():
            print(f"\nProcessing category: {category}")
            for graph_type, graph_data in category_data.items():
                print(f"Processing graph_type: {graph_type}")
                for embedding_type, config_paths in graph_data.items():
                    print(f"Processing embedding_type: {embedding_type} with paths: {config_paths}")
                    # Map the embedding_type to the correct preprocessing group key
                    key = f"{graph_type}_{embedding_type}"
                    if key in preprocessing_groups:
                        preprocessing_groups[key].extend(config_paths)
                        print(f"Added paths to {key}: {config_paths}")
        
        print("\nFinal preprocessing_groups:", json.dumps(preprocessing_groups, indent=2))
        return preprocessing_groups

class TestConfigRunner:
    def __init__(self, config_path: str):
        print(f"Initializing TestConfigRunner with config_path: {config_path}")
        self.config_path = config_path
        print("Creating variations...")
        self.variations = self.create_variations()
        print(f"Created {len(self.variations)} variations")
        print("Splitting config...")
        self.config_structure = VariationType.split_config(self.variations)
        print("Config structure created")
        self.result_manager = ResultManager()
        self.selection = None  # Store selection as class variable
        
        # Default configuration that will be applied to all tests
        self.default_config = {
            "logger": {
                "level": "WARNING"
            },
            "document_range": {
                "enabled": True,
                "document_ids": [1]  # Default document IDs
            },
            "rag": {
                "text_similarity_threshold": 0.3
            },
            "iteration": {
                "loop_retries": 2,
                "pass_threshold": 0.1
            }
        }
        
        # Default prompts for testing
        self.default_prompts = [
            "What programs are there to enhance broadband?",
            "What does the Department of Digital, Culture, Media and Sport do?",
            "What is FTTC?"
        ]

    @staticmethod
    def deep_merge(dict1, dict2):
        """
        Simple merge where dict2 values completely override dict1 values.
        No recursive merging - just straight override.
        """
        merged = dict1.copy()
        merged.update(dict2)
        return merged

    def select_test_types(self):
        # Create checkboxes for each test type
        test_types = {
            "General": widgets.Checkbox(value=False, description='General'),
            "API_LIMITED": widgets.Checkbox(value=False, description='API Limited'),
            "API_HARDWARE_LIMITED": widgets.Checkbox(value=False, description='API & Hardware Limited')
        }
        
        # Create preprocessing controls
        preprocessing_enabled = widgets.Checkbox(value=False, description='Enable Preprocessing')
        preprocessing_type = widgets.SelectMultiple(
            options=['llm_graph_aws_embedding', 'llm_graph_clip_embedding', 
                    'bert_graph_aws_embedding', 'bert_graph_clip_embedding'],
            description='Preprocessing Types:',
            disabled=True,
            layout=widgets.Layout(width='50%', height='100px')
        )
        
        # Create overwrite checkbox
        overwrite_enabled = widgets.Checkbox(value=False, description='Allow Overwrite')
        
        # Create prompts input
        prompts_input = widgets.Textarea(
            value='\n'.join(self.default_prompts),
            description='Test Prompts:',
            layout=widgets.Layout(width='100%', height='100px')
        )
        
        # Link preprocessing checkbox to dropdown
        def on_preprocessing_change(change):
            preprocessing_type.disabled = not change['new']
        preprocessing_enabled.observe(on_preprocessing_change, names='value')
        
        # Create output widget for displaying results
        output = widgets.Output()
        
        def log_message(message):
            with output:
                clear_output()
                print(message)
        
        # Store the widgets in the class variable
        self.selection = {
            'test_types': test_types,
            'preprocessing_enabled': preprocessing_enabled,
            'preprocessing_type': preprocessing_type,
            'overwrite_enabled': overwrite_enabled,
            'prompts_input': prompts_input,
            'output': output
        }
        
        # Display widgets
        display(widgets.VBox([
            widgets.HBox([v for v in test_types.values()]),
            widgets.HBox([preprocessing_enabled, preprocessing_type]),
            overwrite_enabled,
            prompts_input,
            output
        ]))

    def run_selected_tests(self):
        if not self.selection:
            print("Please select test types first")
            return
            
        test_types = self.selection['test_types']
        preprocessing_enabled = self.selection['preprocessing_enabled']
        preprocessing_type = self.selection['preprocessing_type']
        overwrite_enabled = self.selection['overwrite_enabled']
        prompts_input = self.selection['prompts_input']
        output = self.selection['output']
        
        def log_message(message):
            with output:
                clear_output()
                print(message)
        
        selected_types = [k for k, v in test_types.items() if v.value]
        if not selected_types:
            log_message("Please select at least one test type")
            return
        
        # Get the selected configurations
        selected_configs = {k: self.config_structure[k] for k in selected_types}
        
        # Group by preprocessing
        preprocessing_groups = VariationType.group_by_preprocessing(selected_configs)
        log_message(f"Selected test types: {selected_types}\nPreprocessing enabled: {preprocessing_enabled.value}")
        if preprocessing_enabled.value:
            log_message(f"Selected preprocessing types: {preprocessing_type.value}")
        
        # Filter by preprocessing type if enabled
        if preprocessing_enabled.value:
            if not preprocessing_type.value:
                log_message("Please select at least one preprocessing type")
                return
            
            log_message(f"Available preprocessing groups: {list(preprocessing_groups.keys())}")
            log_message(f"Selected preprocessing types: {preprocessing_type.value}")
            
            filtered_groups = {}
            for ptype in preprocessing_type.value:
                if ptype in preprocessing_groups:
                    filtered_groups[ptype] = preprocessing_groups[ptype]
                    log_message(f"Added {ptype} to filtered groups with {len(preprocessing_groups[ptype])} configs")
                else:
                    log_message(f"No configurations found for preprocessing type: {ptype}")
            
            if not filtered_groups:
                log_message("No valid preprocessing configurations found")
                return
            
            preprocessing_groups = filtered_groups
            log_message(f"Final filtered groups: {list(preprocessing_groups.keys())}")
        
        # Get prompts from input
        prompts = [p.strip() for p in prompts_input.value.split('\n') if p.strip()]
        try:
            # Initialize PsycoreTestRunner with timeout
            log_message("Initializing PsycoreTestRunner...")
            import threading
            import time
            
            runner = None
            init_error = None
            
            def init_runner():
                nonlocal runner, init_error
                try:
                    runner = PsycoreTestRunner.PsycoreTestRunner(preprocess=False)
                except Exception as e:
                    init_error = e
            
            # Start initialization in a separate thread
            init_thread = threading.Thread(target=init_runner)
            init_thread.daemon = True
            init_thread.start()
            
            # Wait for initialization with timeout
            timeout = 30  # seconds
            start_time = time.time()
            while init_thread.is_alive():
                if time.time() - start_time > timeout:
                    log_message("Error: PsycoreTestRunner initialization timed out after 30 seconds.\nPlease check your Pinecone credentials and connection.")
                    return
                time.sleep(0.1)
            
            if init_error:
                log_message(f"Error initializing PsycoreTestRunner: {str(init_error)}")
                return
                
            log_message("PsycoreTestRunner initialized successfully")
            
            # Process each group
            for group, configs in preprocessing_groups.items():
                # Flatten the list of lists and remove duplicates
                print(group)
                for i, config_path in enumerate(configs):
                    try:
                        config_name = os.path.basename(config_path)
                        log_message(f"Testing configuration: {config_name}")
                        
                        # Load and merge configuration
                        log_message(f"Loading configuration from {config_name}...")
                        try:
                            with open(config_path, 'r') as f:
                                config = yaml.safe_load(f)
                            if config is None:
                                log_message(f"Warning: Empty or invalid YAML file: {config_name}")
                                continue
                        except yaml.YAMLError as e:
                            log_message(f"Error parsing YAML file {confisg_name}: {str(e)}")
                            continue
                        
                        # Merge with default config
                        merged_config = TestConfigRunner.deep_merge(config, self.default_config)
                        print(merged_config)
                        # Print merged config before preprocessing
                        log_message(f"\nMerged config before preprocessing for {config_name}:")
                        log_message(json.dumps(merged_config, indent=2))
                        
                        # Check if result already exists
                        exists, config_hash = self.result_manager.check_hash_exists(merged_config)
                        if exists and not overwrite_enabled.value:
                            log_message(f"Result already exists for {config_name} (hash: {config_hash}). Skipping...")
                            continue
                        
                        # Update runner configuration
                        log_message(f"Updating runner configuration...")
                        runner.update_config(merged_config, (i == 0 and preprocessing_enabled.value == True))
                        
                        # Run tests
                        log_message(f"Running tests with prompts...")
                        results = runner.evaluate_prompts(prompts)
                        
                        # Map results to prompts
                        prompt_results = {prompt: result for prompt, result in zip(prompts, results)}
                        
                        # Save results
                        log_message(f"Saving results...")
                        self.result_manager.write_result(merged_config, prompt_results)
                        
                        log_message(f"Completed testing: {config_name}")
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"Error processing {os.path.basename(config_path)}:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        print(error_msg)
                        continue
            
            log_message("All tests completed successfully!")
            
        except Exception as e:
            import traceback
            error_msg = f"Critical error during test execution:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            log_message(error_msg)

    def run_test(self):
        print("Starting run_test method")
        self.select_test_types()
        # Create a button to run the tests
        run_button = widgets.Button(description='Run Selected Tests')
        run_button.on_click(lambda b: self.run_selected_tests())
        display(run_button)

    def create_variations(self) -> list[VariationType]:
        variations = []
        base_path = self.config_path
        print(f"Creating variations from base path: {base_path}")
        print(f"Base path exists: {os.path.exists(base_path)}")

        # API_and_Hardware_Intensive variations
        intensive_path = os.path.join(base_path, "API_and_Hardware_Intensive")
        print(f"Checking intensive path: {intensive_path}")
        print(f"Intensive path exists: {os.path.exists(intensive_path)}")
        if os.path.exists(intensive_path):
            # BERT Graph variations
            bert_aws_path = os.path.join(intensive_path, "BERT_Graph_AWS_Embedding")
            print(f"Checking BERT AWS path: {bert_aws_path}")
            print(f"BERT AWS path exists: {os.path.exists(bert_aws_path)}")
            if os.path.exists(bert_aws_path):
                for yaml_file in os.listdir(bert_aws_path):
                    if yaml_file.endswith('.yaml'):
                        full_path = os.path.join(bert_aws_path, yaml_file)
                        print(f"Found BERT AWS config: {full_path}")
                        variations.append(VariationType(bert_graph=True, llm_graph=False, aws_embedding=True, api_limited=True, hardware_limited=True, config_path=full_path))
            
            bert_clip_path = os.path.join(intensive_path, "BERT_Graph_CLIP_Embedding")
            print(f"Checking BERT CLIP path: {bert_clip_path}")
            print(f"BERT CLIP path exists: {os.path.exists(bert_clip_path)}")
            if os.path.exists(bert_clip_path):
                for yaml_file in os.listdir(bert_clip_path):
                    if yaml_file.endswith('.yaml'):
                        full_path = os.path.join(bert_clip_path, yaml_file)
                        print(f"Found BERT CLIP config: {full_path}")
                        variations.append(VariationType(bert_graph=True, llm_graph=False, aws_embedding=False, api_limited=True, hardware_limited=True, config_path=full_path))
            
            # LLM Graph variations
            llm_aws_path = os.path.join(intensive_path, "LLM_Graph_AWS_Embedding")
            print(f"Checking LLM AWS path: {llm_aws_path}")
            print(f"LLM AWS path exists: {os.path.exists(llm_aws_path)}")
            if os.path.exists(llm_aws_path):
                for yaml_file in os.listdir(llm_aws_path):
                    if yaml_file.endswith('.yaml'):
                        full_path = os.path.join(llm_aws_path, yaml_file)
                        print(f"Found LLM AWS config: {full_path}")
                        variations.append(VariationType(bert_graph=False, llm_graph=True, aws_embedding=True, api_limited=True, hardware_limited=True, config_path=full_path))
            
            llm_clip_path = os.path.join(intensive_path, "LLM_Graph_CLIP_Embedding")
            print(f"Checking LLM CLIP path: {llm_clip_path}")
            print(f"LLM CLIP path exists: {os.path.exists(llm_clip_path)}")
            if os.path.exists(llm_clip_path):
                for yaml_file in os.listdir(llm_clip_path):
                    if yaml_file.endswith('.yaml'):
                        full_path = os.path.join(llm_clip_path, yaml_file)
                        print(f"Found LLM CLIP config: {full_path}")
                        variations.append(VariationType(bert_graph=False, llm_graph=True, aws_embedding=False, api_limited=True, hardware_limited=True, config_path=full_path))

        # API_Limited variations (LLM Graph with AWS Embedding)
        api_limited_path = os.path.join(base_path, "API_Limited")
        print(f"Checking API Limited path: {api_limited_path}")
        print(f"API Limited path exists: {os.path.exists(api_limited_path)}")
        if os.path.exists(api_limited_path):
            for yaml_file in os.listdir(api_limited_path):
                if yaml_file.endswith('.yaml'):
                    full_path = os.path.join(api_limited_path, yaml_file)
                    print(f"Found API Limited config: {full_path}")
                    variations.append(VariationType(bert_graph=False, llm_graph=True, aws_embedding=True, api_limited=True, hardware_limited=False, config_path=full_path))

        # General Models variations (LLM Graph with AWS Embedding)
        general_path = os.path.join(base_path, "General_Models")
        print(f"Checking General Models path: {general_path}")
        print(f"General Models path exists: {os.path.exists(general_path)}")
        if os.path.exists(general_path):
            for yaml_file in os.listdir(general_path):
                if yaml_file.endswith('.yaml'):
                    full_path = os.path.join(general_path, yaml_file)
                    print(f"Found General Models config: {full_path}")
                    variations.append(VariationType(bert_graph=False, llm_graph=True, aws_embedding=True, api_limited=False, hardware_limited=False, config_path=full_path))

        print(f"Total variations found: {len(variations)}")
        return variations


print("Creating TestConfigRunner instance...")
# Get the current working directory
current_dir = os.getcwd()
config_path = os.path.join(current_dir, "config_variations")
print(f"Using config path: {config_path}")
print(f"Config path exists: {os.path.exists(config_path)}")
runner = TestConfigRunner(config_path)
print("TestConfigRunner instance created")
runner.select_test_types()

if input("Run tests? (y/n)") == "y":
    runner.run_selected_tests()