# 🌍 Big Data Mining Project: PageRank & Hadoop Streaming  

## 📜 Overview  
This project explores **Big Data Processing using Hadoop & MapReduce**, with a focus on **PageRank computation, Bloom Filters, and Hadoop Streaming**. The tasks involve:  
- **MapReduce for Large-Scale Data Processing**  
- **Implementing a Bloom Filter for Set Membership Testing**  
- **PageRank Calculation for Directed Graphs**  
- **Handling Dead-End Nodes in PageRank**  

## 🚀 Part 1: Hadoop Streaming  
💡 **Task**: Run a Hadoop Streaming job to process data from multiple tables.  

## 🔍 Part 2: Bloom Filter Implementation
💡 **Task**: Bloom Filter Setup
- SHA-256 Hashing used for element insertion.
- Two hash functions applied to insert values.
- Tested false positives with non-inserted values.

## 📈 Part 3: PageRank Computation
💡 **Task**: Network Graph Analysis
- Initial PageRank assigned as 1/N (N = total nodes).
- Matrix-based iteration until convergence.
- Compared manual vs. Python NetworkX library output.
- 📌 Convergence Achieved in 6 Iterations

## ❗ Part 4: Handling Dead-End Nodes in PageRank
💡 **Task**: Approach 1 - Random Teleporting (Random Walk)
- Dead-end nodes get equal probability distribution (1/N).
- Ensures stochastic matrix property (column sums to 1).
- Python implementation matches manual calculations.
💡 **Task**: Approach 2 - Removing Dead Ends Iteratively
- Recursively remove dead-end nodes and compute PageRank for the remaining graph.
- Final PageRanks reassigned based on original network structure.

## 📘 Part 5: Recursive Dead-End Handling in a Chain Graph
💡 **Scenario**: Dead-end nodes in a sequential chain.

📌 **Observation**:
- Origin node with self-loop retains PageRank = 1.
- Dead-end nodes inherit PageRank from previous nodes.
- PageRank follows an exponential decay pattern.

## 🔢 Part 6: PageRank on Large Graphs (Stanford Web Dataset)

## 🚀 Technologies Used
🛠 **Big Data Frameworks**:
- Hadoop (MapReduce, HDFS)
- Python (NetworkX, NumPy, Pandas)
