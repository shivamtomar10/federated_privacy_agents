from system.orchestrator import FederatedOrchestrator


DATASETS = {
    "india": "data/hospital_india.csv",
    "usa": "data/hospital_usa.csv",
    "japan": "data/hospital_japan.csv",
    "uk": "data/hospital_uk.csv"
}


if __name__ == "__main__":
    orchestrator = FederatedOrchestrator(DATASETS)
    orchestrator.run()

