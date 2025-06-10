from typing import Dict, List, Any, Set, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SubchainGenerator:
    def __init__(self, config: Dict):
        """
        Initialize the subchain generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.subchain_configs = config.get("subchain_generation", {}).get("configurations", [])
        self.deduplicate = config.get("subchain_generation", {}).get("deduplicate_subchains", True)
        self.enabled = config.get("subchain_generation", {}).get("enabled", False)
        
        # Setup output paths
        self.output_dir = Path(config["paths"]["output_dir"])
        self.subchains_dir = self.output_dir / "subchains"
        self.subchains_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SubchainGenerator - Enabled: {self.enabled}")
        if self.enabled:
            enabled_configs = [c for c in self.subchain_configs if c.get("enabled", True)]
            logger.info(f"Found {len(enabled_configs)} enabled subchain configurations")
            for i, config in enumerate(enabled_configs):
                logger.info(f"  Config {i+1}: window_size={config['window_size']}, shift={config['shift']}")
        
    def generate_subchains(self, chain: str) -> Dict[str, List[str]]:
        """
        Generate subchains for a given chain using all enabled configurations
        
        Args:
            chain: Input cleaned chain (e.g., "cat1>cat2>cat3>cat4")
            
        Returns:
            Dictionary mapping config identifiers to lists of subchains
        """
        if not self.enabled:
            return {}
            
        logger.debug(f"Generating subchains for: {chain}")
        
        result = {}
        
        for config in self.subchain_configs:
            if not config.get("enabled", True):
                continue
                
            window_size = config["window_size"]
            shift = config["shift"]
            config_id = f"w{window_size}_s{shift}"
            
            subchains = self._extract_subchains_with_config(chain, window_size, shift)
            result[config_id] = subchains
            
            logger.debug(f"Config {config_id} generated {len(subchains)} subchains from: {chain}")
            
        return result
        
    def _extract_subchains_with_config(self, chain: str, window_size: int, shift: int) -> List[str]:
        """
        Extract subchains using sliding window approach
        
        Args:
            chain: Input chain (e.g., "cat1>cat2>cat3>cat4")
            window_size: Size of the sliding window
            shift: Step size for sliding window
            
        Returns:
            List of subchains
        """
        # Split chain into categories
        categories = chain.split('>')
        categories = [cat.strip() for cat in categories if cat.strip()]
        
        if len(categories) < window_size:
            # If chain is shorter than window, return the whole chain
            return [chain] if categories else []
        
        subchains = []
        
        # Sliding window extraction
        for i in range(0, len(categories) - window_size + 1, shift):
            window_categories = categories[i:i + window_size]
            subchain = '>'.join(window_categories)
            subchains.append(subchain)
            
        return subchains
        
    def generate_all_subchains(self, cleaned_chains: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate subchains for all chains and create comprehensive mappings
        
        Args:
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            Dictionary containing all subchain data and mappings
        """
        if not self.enabled:
            logger.info("Subchain generation is disabled")
            return {
                "enabled": False,
                "subchain_mappings": {},
                "deduplicated_subchains": [],
                "config_used": [],
                "stats": {"total_subchains": 0, "unique_subchains": 0}
            }
            
        logger.info(f"Generating subchains for {len(cleaned_chains)} cleaned chains")
        
        # Store subchains for each chain
        chain_subchains = {}
        all_subchains_by_config = {}
        
        # Process each cleaned chain
        for original_chain, cleaned_chain in cleaned_chains.items():
            subchains_by_config = self.generate_subchains(cleaned_chain)
            chain_subchains[cleaned_chain] = subchains_by_config
            
            # Collect all subchains by configuration
            for config_id, subchains in subchains_by_config.items():
                if config_id not in all_subchains_by_config:
                    all_subchains_by_config[config_id] = []
                all_subchains_by_config[config_id].extend(subchains)
        
        # Deduplicate subchains if enabled
        deduplicated_subchains = []
        if self.deduplicate:
            all_subchains = []
            for subchains in all_subchains_by_config.values():
                all_subchains.extend(subchains)
            deduplicated_subchains = list(set(all_subchains))
            logger.info(f"Deduplicated {len(all_subchains)} total subchains to {len(deduplicated_subchains)} unique subchains")
        else:
            for subchains in all_subchains_by_config.values():
                deduplicated_subchains.extend(subchains)
            logger.info(f"No deduplication - total {len(deduplicated_subchains)} subchains")
        
        # Create reverse mappings for easier lookup
        reverse_mappings = self._create_reverse_mappings(chain_subchains, cleaned_chains)
        
        # Prepare result
        # 
        result = {
            "enabled": True,
            "subchain_mappings": chain_subchains,
            "deduplicated_subchains": deduplicated_subchains,
            "all_subchains_by_config": all_subchains_by_config,
            "reverse_mappings": reverse_mappings,
            "config_used": [c for c in self.subchain_configs if c.get("enabled", True)],
            "stats": {
                "total_chains_processed": len(cleaned_chains),
                "total_subchains": sum(len(subs) for config_subs in all_subchains_by_config.values() for subs in [config_subs]),
                "unique_subchains": len(deduplicated_subchains),
                "configs_used": len([c for c in self.subchain_configs if c.get("enabled", True)])
            }
        }
        
        # Save intermediate results
        self._save_subchain_data(result)
        
        logger.info(f"Generated subchains successfully:")
        logger.info(f"  - Total chains processed: {result['stats']['total_chains_processed']}")
        logger.info(f"  - Total subchains generated: {result['stats']['total_subchains']}")
        logger.info(f"  - Unique subchains: {result['stats']['unique_subchains']}")
        logger.info(f"  - Configurations used: {result['stats']['configs_used']}")
        
        return result
        
    def _create_reverse_mappings(self, chain_subchains: Dict[str, Dict[str, List[str]]], cleaned_chains: Dict[str, str]) -> Dict[str, Any]:
        """
        Create reverse mappings from subchains back to original chains
        
        Args:
            chain_subchains: Mapping of chains to their subchains
            cleaned_chains: Mapping of original to cleaned chains
            
        Returns:
            Dictionary containing reverse mappings
        """
        subchain_to_chains = {}
        subchain_to_original_chains = {}
        
        # Create reverse mapping from cleaned chains to original chains
        cleaned_to_original = {v: k for k, v in cleaned_chains.items()}
        
        for cleaned_chain, subchains_by_config in chain_subchains.items():
            original_chain = cleaned_to_original.get(cleaned_chain, cleaned_chain)
            
            for config_id, subchains in subchains_by_config.items():
                for subchain in subchains:
                    if subchain not in subchain_to_chains:
                        subchain_to_chains[subchain] = []
                        subchain_to_original_chains[subchain] = []
                    
                    if cleaned_chain not in subchain_to_chains[subchain]:
                        subchain_to_chains[subchain].append(cleaned_chain)
                    
                    if original_chain not in subchain_to_original_chains[subchain]:
                        subchain_to_original_chains[subchain].append(original_chain)
        
        return {
            "subchain_to_cleaned_chains": subchain_to_chains,
            "subchain_to_original_chains": subchain_to_original_chains
        }
        
    def _save_subchain_data(self, subchain_data: Dict[str, Any]) -> None:
        """
        Save subchain data to files
        
        Args:
            subchain_data: Complete subchain data dictionary
        """
        try:
            # Save subchain mappings
            mappings_path = self.subchains_dir / "subchain_mappings.json"
            with open(mappings_path, 'w') as f:
                json.dump(subchain_data["subchain_mappings"], f, indent=2)
            
            # Save deduplicated subchains
            deduplicated_path = self.subchains_dir / "deduplicated_subchains.json"
            with open(deduplicated_path, 'w') as f:
                json.dump(subchain_data["deduplicated_subchains"], f, indent=2)
            
            # Save configuration used
            config_path = self.subchains_dir / "subchain_configs.json"
            config_data = {
                "enabled": subchain_data["enabled"],
                "deduplicate_subchains": self.deduplicate,
                "configurations": subchain_data["config_used"],
                "stats": subchain_data["stats"]
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Save reverse mappings
            reverse_mappings_path = self.subchains_dir / "reverse_mappings.json"
            with open(reverse_mappings_path, 'w') as f:
                json.dump(subchain_data["reverse_mappings"], f, indent=2)
            
            logger.info(f"Saved subchain data to {self.subchains_dir}")
            
        except Exception as e:
            logger.error(f"Error saving subchain data: {e}")
            raise
            
    def get_subchains_for_factsheet_mapping(self, subchain_data: Dict[str, Any], cleaned_chains: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Create mapping from subchains to original chains for factsheet association
        
        Args:
            subchain_data: Complete subchain data
            cleaned_chains: Mapping of original to cleaned chains
            
        Returns:
            Dictionary mapping each subchain to list of original chains it belongs to
        """
        if not subchain_data.get("enabled", False):
            return {}
            
        reverse_mappings = subchain_data.get("reverse_mappings", {})
        subchain_to_original = reverse_mappings.get("subchain_to_original_chains", {})
        
        logger.info(f"Created factsheet mapping for {len(subchain_to_original)} subchains")
        
        return subchain_to_original
        
    def create_company_subchains_mapping(
        self,
        subchain_data: Dict[str, Any],
        company_chains: Dict[str, List[str]],
        cleaned_chains: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Create mapping from companies to subchains (similar to company_chains but for subchains)
        
        Args:
            subchain_data: Complete subchain data from generate_all_subchains
            company_chains: Original mapping of companies to chains
            cleaned_chains: Mapping of original to cleaned chains
            
        Returns:
            Dictionary mapping company IDs to lists of subchains
        """
        if not subchain_data.get("enabled", False):
            logger.info("Subchain generation disabled - returning empty company_subchains mapping")
            return {}
            
        logger.info("Creating company-subchains mapping for overlap calculation")
        
        # Get reverse mapping from cleaned chains to original chains
        cleaned_to_original = {v: k for k, v in cleaned_chains.items()}
        
        # Get subchain mappings
        subchain_mappings = subchain_data.get("subchain_mappings", {})
        
        company_subchains = {}
        
        # For each company and their chains
        for company_id, original_chains in company_chains.items():
            company_subchains[company_id] = []
            
            # For each chain belonging to this company
            for original_chain in original_chains:
                # Get the cleaned version of this chain
                cleaned_chain = cleaned_chains.get(original_chain)
                if not cleaned_chain:
                    continue
                    
                # Get subchains for this cleaned chain
                chain_subchains = subchain_mappings.get(cleaned_chain, {})
                
                # Add all subchains from all configurations
                for config_id, subchains in chain_subchains.items():
                    company_subchains[company_id].extend(subchains)
            
            # Remove duplicates
            company_subchains[company_id] = list(set(company_subchains[company_id]))
        
        # Filter out companies with no subchains
        company_subchains = {k: v for k, v in company_subchains.items() if v}
        
        total_subchains = sum(len(subchains) for subchains in company_subchains.values())
        logger.info(f"Created company-subchains mapping: {len(company_subchains)} companies mapped to {total_subchains} total subchains")
        
        return company_subchains 