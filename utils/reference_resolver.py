class ReferenceResolver:

    KEYWORDS = [
        "above company", "previous company", "that company",
        "above stock", "previous stock", "that stock",
        "same company", "same stock", "it", "that"
    ]

    def __init__(self, memory_manager):
        self.memory = memory_manager

    def resolve(self, user_id: str, query: str) -> str:
        q = query.lower()
        ent = self.memory.get_all_entities(user_id)

        last_company = ent.get("last_company")
        last_ticker = ent.get("last_ticker")

        if not last_company and not last_ticker:
            return query

        for kw in self.KEYWORDS:
            if kw in q:
                if last_company:
                    q = q.replace(kw, last_company.lower())
                elif last_ticker:
                    q = q.replace(kw, last_ticker.lower())

        return q