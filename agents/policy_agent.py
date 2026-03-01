class PolicyAgent:
    def decide(self, column, country):
        col = column.lower()

        if country == "germany":
            if "zip" in col or "address" in col:
                return "HIGH"

        if country == "india":
            if "aadhaar" in col or "phone" in col:
                return "HIGH"

        if country == "usa":
            if "ssn" in col:
                return "HIGH"
            if any(k in col for k in ["name", "email", "address"]):
                return "HIGH"

        return "LOW"

