[LICENSE]: https://github.com/atomgptlab/jarvis/blob/master/LICENSE.rst


<style>
* {
  box-sizing: border-box;
}

.column {
  float: left;
  width: 33.33%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}

* {
  box-sizing: border-box;
}

body {
  font-family: Arial, Helvetica, sans-serif;
}

/* Float four columns side by side */
.column {
  float: left;
  width: 25%;
  padding: 0 10px;
}

/* Remove extra left and right margins, due to padding */
.row {margin: 0 -5px;}

/* Clear floats after the columns */
.row:after {
  content: "";
  display: table;
  clear: both;
}

/* Responsive columns */
@media screen and (max-width: 600px) {
  .column {
    width: 100%;
    display: block;
    margin-bottom: 20px;
  }
}

/* Style the counter cards */
.card {
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
  padding: 20px;
  text-align: center;
  background-color: #f1f1f1;
  margin-bottom: 20px;
}

</style>


<h1 style="text-align:center;">Explore State-of-the-Art Materials Design Methods</h1>
<div class="row">
  <div class="column">
    <div class="card">
<h3>Artificial intelligence (AI)</h3><p>Contributions: 1067</p>

      <a href="./AI" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Electronic Struct. (ES)</h3><p>Contributions: 741</p>
      <a href="./ES" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Force-field (FF)/potentials</h3><p>Contributions 282</p>
      <a href="./FF" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Quantum Comput. (QC) </h3><p>Contributions: 6</p>
      <a href="./QC" >See All Benchmarks</a>
    </div>
  </div>

</div>
<div class="row">

  <div class="column">
    <div class="card">
<h3>Experiments (EXP)</h3><p>Contributions: 25</p>
      <a href="./EXP" >See All Benchmarks</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
      <h3>Example Notebooks</h3><p>Notebooks: 16</p>
      <a href="./notebooks" >See All Notebooks</a>
    </div>
  </div>


  <div class="column">
    <div class="card">
<h3>Methodologies</h3><p>Available Methods:504</p>
      <a href="https://github.com/usnistgov/jarvis_leaderboard/tree/main/jarvis_leaderboard/contributions">Learn more</a>
    </div>
  </div>

  <div class="column">
    <div class="card">
<h3>Contribution Guide</h3><p>Contributors: <a href="https://github.com/atomgptlab/jarvis_leaderboard/graphs/contributors" >26</a></p>
      <a href="./guide/guide_short" >Learn more</a>
    </div>
  </div>

</div>

---

!!! Reference

    [JARVIS-Leaderboard: a large scale benchmark of materials design methods, Nature npj Computational Materials volume 10, 93 (2024)](https://www.nature.com/articles/s41524-024-01259-w)



<div class="row">
<div class="column">
<h1 id="table-of-contents">Table of Contents</h1>
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#example">Example benchmarks</a></li>
<li><a href="#license">License</a></li>
<li><a href="#help">Help</a></li>
</ul> 
</div>
<!--
<div class="column">
<iframe width="420" align="center" height="215" src="https://www.youtube.com/embed/QDx3jSIwpMo?autoplay=1&mute=1" frameborder="0" allowfullscreen></iframe>
</div>
-->
</div>
<a name="intro"></a>
# Introduction

This project provides benchmark performances of various methods for materials science applications using the datasets available in [JARVIS-Tools databases](https://pages.nist.gov/jarvis/databases/). Some of the categories are: [Artificial Intelligence (AI)](./AI), [Electronic Structure (ES)](./ES), [Force-field (FF)](./FF), [Quantum Computation (QC)](./QC) and [Experiments (EXP)](./EXP). A variety of properties are included in the benchmark.

Typically, codes are kept in platforms like GitHub/GitLab, and data is stored in repositories like Zenodo/Figshare/NIST Materials Data. We recommend keeping the benchmarks in the JARVIS-Leaderboard to enhance reproducibility and transparency.
In addition to prediction results, we aim to capture the underlying software, hardware, and instrumental frameworks to enhance reproducibility. This project is a part of the [NIST-JARVIS](https://jarvis.nist.gov) infrastructure.
As a minimum check, we test rebuilding of the leaderboard and installations of software using GitHub actions.

![Leaderboard actions](https://github.com/atomgptlab/jarvis_leaderboard/actions/workflows/test_build.yml/badge.svg)
![Leaderboard AI installation actions](https://github.com/atomgptlab/jarvis_leaderboard/actions/workflows/install_ai.yml/badge.svg)
![Leaderboard ES installation actions](https://github.com/atomgptlab/jarvis_leaderboard/actions/workflows/install_es.yml/badge.svg)
![Leaderboard FF installation actions](https://github.com/atomgptlab/jarvis_leaderboard/actions/workflows/install_ff.yml/badge.svg)
![Leaderboard QC installation actions](https://github.com/atomgptlab/jarvis_leaderboard/actions/workflows/install_qc.yml/badge.svg)


<!--number_of_benchmarks--> - Number of benchmarks: 14

<!--number_of_contributions--> - Number of contributions: 42

<!--number_of_datapoints--> - Number of datapoints: 65177

<!-- [Learn how to add benchmarks below](#add) -->
<!-- <p style="text-align:center;"><img align="middle" src="https://www.ctcms.nist.gov/~knc6/images/logo/jarvis-mission.png"  width="40%" height="20%"></p>-->




A brief summary table is given below:


<!--summary_table--><table style="width:100%" id="j_table"><thead><td>Category/Sub-cat.</td><td>SinglePropertyPrediction</td><td>SinglePropertyClass</td><td>MLFF</td><td>TextClass</td><td>TokenClass</td><td>TextSummary</td><td>TextGen</td><td>AtomGen</td><td>ImageClass</td><td>Spectra</td><td>EigenSolver</td><tr><td>AI</td><td><a href="./AI/SinglePropertyPrediction" target="_blank">702</a></td><td><a href="./AI/SinglePropertyClass" target="_blank">21</a></td><td><a href="./AI/MLFF" target="_blank">266</a></td><td><a href="./AI/TextClass" target="_blank">28</a></td><td><a href="./AI/TokenClass" target="_blank">1</a></td><td><a href="./AI/TextSummary" target="_blank">1</a></td><td><a href="./AI/TextGen" target="_blank">3</a></td><td><a href="./AI/AtomGen" target="_blank">42</a></td><td><a href="./AI/ImageClass" target="_blank">2</a></td><td><a href="./AI/Spectra" target="_blank">1</a></td><td>-</td><tr><tr><td>ES</td><td><a href="./ES/SinglePropertyPrediction" target="_blank">731</a></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td><a href="./ES/Spectra" target="_blank">10</a></td><td>-</td><tr><tr><td>FF</td><td><a href="./FF/SinglePropertyPrediction" target="_blank">282</a></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><tr><tr><td>QC</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td><a href="./QC/EigenSolver" target="_blank">6</a></td><tr><tr><td>EXP</td><td><a href="./EXP/SinglePropertyPrediction" target="_blank">7</a></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td><a href="./EXP/Spectra" target="_blank">18</a></td><td>-</td><tr></table>





<a name="example"></a>
# Example benchmarks
Click on the entries in the Benchmark column. You'll be able to see performance comparison, methods available for each benchmark, CSV file submitted for the contribution, JSON file for ground trutch data, run.sh script for running the method and Info for metadata associated with the method.

<!--table_content--><table style="width:100%" id="j_table"><thead><tr><th>Category</th><th>Sub-category</th><th>Benchmark</th><th>Method</th><th>Metric</th><th>Score</th><th>Team</th><th>Dataset</th><th>Size</th></tr></thead><!--table_content--></table>









       
<a name="help"></a>
# Help

   If you have a question/suggestion, raise a [GitHub issue](https://github.com/atomgptlab/jarvis_leaderboard/issues) or submit a [Google form](https://forms.gle/giDEnfNmkaU5BhBw9) request.


<a name="license"></a>
# License

   This template is served under the NIST license.  
   Read the [LICENSE] file for more info.