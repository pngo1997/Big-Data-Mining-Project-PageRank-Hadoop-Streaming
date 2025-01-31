# 🌍 Big Data Mining Project: PageRank & Hadoop Streaming  

## 📜 Overview  
This project explores **Big Data Processing using Hadoop & MapReduce**, with a focus on **PageRank computation, Bloom Filters, and Hadoop Streaming**. The tasks involve:  
- **MapReduce for Large-Scale Data Processing**  
- **Implementing a Bloom Filter for Set Membership Testing**  
- **PageRank Calculation for Directed Graphs**  
- **Handling Dead-End Nodes in PageRank**  

📌 **Course**: CSC 555 - Big Data Mining  
📌 **Date**: November 10, 2023  
📌 **Student**: Mai Ngo  

---

## 🚀 Part 1: Hadoop Streaming  

💡 **Task**: Run a Hadoop Streaming job to process data from multiple tables.  

### **Hadoop Command Execution**  
```bash
time hadoop jar hadoop-streaming-2.6.4.jar \
  -input /data/lineorder.tbl,/data/dwdate.tbl \
  -output /data/output1 \
  -mapper HW4_mapper.py \
  -reducer HW4_reducer.py \
  -file HW4_mapper.py \
  -file HW4_reducer.py
