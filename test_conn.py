from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_jp4oE_DR9RLG3MGUfuX8Cdqt8DDgVNLtZQuue35DjRBLX1P8JkAPwJmcxXP9NvaQ88aSH")
index = pc.Index("trading-bot")  # index name, correct
print("Before cleanup:", index.describe_index_stats())

# Delete default namespace
index.delete(delete_all=True, namespace="")

# Delete 'text' namespace
index.delete(delete_all=True, namespace="text")

print("After cleanup:", index.describe_index_stats())
print("✔ All Pinecone namespaces cleared.")