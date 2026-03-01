class AttackDetector:
    def inspect(self, aggregated_data):
        if aggregated_data.isnull().any():
            print("⚠️ Possible poisoning detected")
        else:
            print("✅ No attack detected")

