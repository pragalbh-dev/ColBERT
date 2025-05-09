from synthetic_data_generation import SyntheticDataGenerator

if __name__ == "__main__":
    # Configure these parameters as needed
    QUERIES_PER_ASPECT = 100  # Number of queries to generate per aspect
    TOP_K_SEARCH = 30         # Number of search results to fetch per query
    
    print("Starting synthetic data generation process...")
    generator = SyntheticDataGenerator()
    triplets = generator.curate_dataset(
        queries_per_aspect=QUERIES_PER_ASPECT, 
        top_k_search=TOP_K_SEARCH
    )
    
    print(f"Data generation complete! Generated {len(triplets)} triplets.")
    print(f"Data saved to the 'data' directory.") 