import os,sys,json
import yaml
from ipywidgets import widgets
from IPython.display import display, clear_output
from tqdm import tqdm
from discord_webhook import DiscordWebhook

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
    def __init__(self, config_path: str, discord_webhook_url: str = None):
        print(f"Initializing TestConfigRunner with config_path: {config_path}")
        self.config_path = config_path
        self.discord_webhook_url = discord_webhook_url
        if discord_webhook_url:
            self.discord_webhook = DiscordWebhook(discord_webhook_url)
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
                "document_ids": [1, 15, 34, 4, 62, 63, 23, 44, 58, 67]  # Default document IDs
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
        self.default_prompts = ["How did BDUK classify premises as White, Grey, Black, or Under Review in the Norfolk Public Review, and what implications did these classifications have for eligibility for government subsidy?","How has the UK government's strategy for gigabit-capable broadband evolved from the Superfast Broadband Programme to Project Gigabit, and what lessons have been incorporated into the newer interventions?", "What were the key limitations encountered during the OMR and public review processes as identified in the early process evaluation of the Gigabit Infrastructure Subsidy intervention?","How have different procurement models (e.g., regional vs. local suppliers) impacted the speed and effectiveness of broadband infrastructure delivery in rural areas?","What role does passive infrastructure sharing (e.g. ducts and poles) play in the UK's plan for nationwide full fibre deployment, and how does this relate to regulatory goals?"]

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
            disabled=False,
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
        
        def log_message(message, discord_prefix="ℹ️"):
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
        
        # Display widgets in a more organized layout
        display(widgets.VBox([
            widgets.HTML("<h3>Test Types</h3>"),
            widgets.HBox([v for v in test_types.values()]),
            widgets.HTML("<h3>Preprocessing Options</h3>"),
            widgets.HBox([preprocessing_enabled, preprocessing_type]),
            widgets.HTML("<h3>Other Options</h3>"),
            overwrite_enabled,
            widgets.HTML("<h3>Test Prompts</h3>"),
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
        
        def log_message(message, discord_prefix="ℹ️"):
            with output:
                clear_output()
                print(message)
            
            # Send to Discord if webhook is configured
            if hasattr(self, 'discord_webhook'):
                try:
                    # Split long messages into chunks of 2000 characters (Discord's limit)
                    chunks = [message[i:i+2000] for i in range(0, len(message), 2000)]
                    for chunk in chunks:
                        self.discord_webhook.send_message(
                            content=f"{discord_prefix} {chunk}",
                            username="Psycore Test Runner"
                        )
                except Exception as e:
                    print(f"Failed to send Discord notification: {e}")
                
        def send_discord_notification(message: str, prefix="ℹ️"):
            if hasattr(self, 'discord_webhook'):
                try:
                    self.discord_webhook.send_message(
                        content=f"{prefix} {message}",
                        username="Psycore Test Runner"
                    )
                except Exception as e:
                    print(f"Failed to send Discord notification: {e}")

        selected_types = [k for k, v in test_types.items() if v.value]
        if not selected_types:
            log_message("Please select at least one test type", "⚠️")
            return
        
        # Get the selected configurations
        selected_configs = {k: self.config_structure[k] for k in selected_types}
        
        # Group by preprocessing
        preprocessing_groups = VariationType.group_by_preprocessing(selected_configs)
        log_message(f"Selected test types: {selected_types}\nPreprocessing enabled: {preprocessing_enabled.value}")
        
        # Get prompts from input
        prompts = [p.strip() for p in prompts_input.value.split('\n') if p.strip()]
        
        # Send initial Discord notification about test configuration
        start_message = "🚀 Starting new test run with configuration:\n"
        start_message += f"• Test Types: {', '.join(selected_types)}\n"
        start_message += f"• Preprocessing Enabled: {preprocessing_enabled.value}\n"
        if preprocessing_enabled.value:
            start_message += f"• Preprocessing Types: {', '.join(preprocessing_type.value)}\n"
        start_message += f"• Overwrite Enabled: {overwrite_enabled.value}\n"
        start_message += f"• Number of Prompts: {len(prompts)}\n"
        start_message += "• Prompts:\n"
        for i, prompt in enumerate(prompts, 1):
            start_message += f"  {i}. {prompt[:100]}...\n" if len(prompt) > 100 else f"  {i}. {prompt}\n"
        log_message(start_message, "🚀")
        
        # If preprocessing is enabled, we need to select preprocessing types
        if preprocessing_enabled.value:
            if not preprocessing_type.value:
                log_message("Please select at least one preprocessing type", "⚠️")
                return
            
            log_message(f"Selected preprocessing types: {preprocessing_type.value}")
        
        try:
            # Initialize PsycoreTestRunner with timeout
            log_message("Initializing PsycoreTestRunner...", "⚙️")
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
                    log_message("Error: PsycoreTestRunner initialization timed out after 30 seconds.\nPlease check your Pinecone credentials and connection.", "❌")
                    return
                time.sleep(0.1)
            
            if init_error:
                log_message(f"Error initializing PsycoreTestRunner: {str(init_error)}", "❌")
                return
                
            log_message("PsycoreTestRunner initialized successfully", "✅")
            
            # Track the last used graph type and embedding type
            last_graph_type = None
            last_embedding_type = None
            
            # Process each group
            for group, configs in preprocessing_groups.items():
                # Skip groups that don't match selected preprocessing types if preprocessing is enabled
                if preprocessing_enabled.value and group not in preprocessing_type.value:
                    continue
                    
                # Extract current graph type and embedding type from group name
                # Group name format is like "llm_graph_aws_embedding"
                parts = group.split('_')
                current_graph_type = parts[0]  # llm or bert
                current_embedding_type = parts[2]  # aws or clip
                
                # Send notification about current group
                group_message = f"📁 Processing group: {group}\n"
                group_message += f"• Graph Type: {current_graph_type}\n"
                group_message += f"• Embedding Type: {current_embedding_type}\n"
                group_message += f"• Number of configs: {len(configs)}"
                log_message(group_message, "📁")
                
                # Determine if preprocessing is needed
                should_preprocess = preprocessing_enabled.value and (
                    last_graph_type is None or  # First run
                    last_graph_type != current_graph_type or  # Graph type changed
                    last_embedding_type != current_embedding_type  # Embedding type changed
                )
                
                # Update last used types
                last_graph_type = current_graph_type
                last_embedding_type = current_embedding_type
                
                # Process configs in batches of 5
                for i in range(0, len(configs), 5):
                    batch = configs[i:i+5]
                    batch_message = f"📋 Processing batch {i//5 + 1} of {(len(configs) + 4)//5}:\n"
                    for config_path in batch:
                        batch_message += f"• {os.path.basename(config_path)}\n"
                    log_message(batch_message, "📋")
                    
                    for config_path in batch:
                        try:
                            config_name = os.path.basename(config_path)
                            log_message(f"Testing configuration: {config_name}", "⚙️")
                            
                            # Load and merge configuration
                            log_message(f"Loading configuration from {config_name}...", "📄")
                            try:
                                with open(config_path, 'r') as f:
                                    config = yaml.safe_load(f)
                                if config is None:
                                    log_message(f"Warning: Empty or invalid YAML file: {config_name}", "⚠️")
                                    continue
                            except yaml.YAMLError as e:
                                log_message(f"Error parsing YAML file {config_name}: {str(e)}", "❌")
                                continue
                            
                            # Merge with default config
                            merged_config = TestConfigRunner.deep_merge(config, self.default_config)
                            print(merged_config)
                            # Print merged config before preprocessing
                            log_message(f"\nMerged config before preprocessing for {config_name}:", "⚙️")
                            log_message(json.dumps(merged_config, indent=2), "📄")
                            
                            # Check if result already exists
                            exists, config_hash = self.result_manager.check_hash_exists(merged_config)
                            if exists and not overwrite_enabled.value:
                                log_message(f"Result already exists for {config_name} (hash: {config_hash}). Skipping...", "⏭️")
                                continue
                            # Update runner configuration
                            log_message(f"Updating runner configuration...", "⚙️")
                            # Only do preprocessing if it's needed and this is the first config in its group
                            if (should_preprocess and i == 0):
                                log_message(f"Updating runner configuration with True", "🔄")
                                log_message(f"Updating runner configuration where {current_embedding_type} Embedding and {current_graph_type} Graph are used", "🔄")
                                runner.update_config(merged_config, True)
                            else:
                                runner.update_config(merged_config, False)
                            
                            # Run tests
                            log_message(f"Running tests with prompts...", "▶️")
                            results = runner.evaluate_prompts(prompts)
                            
                            # Map results to prompts
                            prompt_results = {prompt: result for prompt, result in zip(prompts, results)}
                            
                            # Save results
                            log_message(f"Saving results...", "💾")
                            self.result_manager.write_result(merged_config, prompt_results)
                            
                            log_message(f"Completed testing: {config_name}", "✅")
                            
                        except Exception as e:
                            import traceback
                            error_msg = f"Error processing {os.path.basename(config_path)}:\n{str(e)}"
                            print(error_msg)
                            log_message(error_msg, "❌")
                            continue
            
            log_message("All tests completed successfully!", "✅")
            
        except Exception as e:
            import traceback
            error_msg = f"Critical error during test execution:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            log_message(error_msg, "❌")
            raise e

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
runner = TestConfigRunner(config_path, discord_webhook_url="https://discord.com/api/webhooks/")
print("TestConfigRunner instance created")
runner.select_test_types()

if input("Run tests? (y/n)") == "y":
    runner.run_selected_tests()