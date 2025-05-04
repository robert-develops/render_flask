document.addEventListener("DOMContentLoaded", function () {
  // Benchmark setup
  document
    .getElementById("setup-benchmarks")
    .addEventListener("click", setupBenchmarks);

  // Observation setup
  document
    .getElementById("setup-observations")
    .addEventListener("click", setupObservations);

  // Compute button
  document
    .getElementById("compute-btn")
    .addEventListener("click", computeAdjustment);
});

function setupBenchmarks() {
  const numBenchmarks = parseInt(
    document.getElementById("num-benchmarks").value
  );
  const container = document.getElementById("benchmark-inputs");
  container.innerHTML = "";

  for (let i = 1; i <= numBenchmarks; i++) {
    const div = document.createElement("div");
    div.className = "row benchmark-input";
    div.innerHTML = `
            <div class="col-md-6">
                <label class="form-label">Benchmark B${i} Elevation (m):</label>
                <input type="number" step="0.001" class="form-control benchmark-elevation" required>
            </div>
        `;
    container.appendChild(div);
  }
}

function setupObservations() {
  const numObservations = parseInt(
    document.getElementById("num-observations").value
  );
  const numParameters = parseInt(
    document.getElementById("num-parameters").value
  );
  const container = document.getElementById("observation-inputs");
  container.innerHTML = "";

  // Create observation inputs
  for (let i = 1; i <= numObservations; i++) {
    const div = document.createElement("div");
    div.className = "row observation-input";
    div.innerHTML = `
            <div class="col-md-4">
                <label class="form-label">Observation #${i}:</label>
                <input type="number" class="form-control obs-no" value="${i}" readonly>
            </div>
            <div class="col-md-4">
                <label class="form-label">Distance (Km):</label>
                <input type="number" step="0.001" class="form-control obs-distance" required>
            </div>
            <div class="col-md-4">
                <label class="form-label">Height Difference (m):</label>
                <input type="number" step="0.001" class="form-control obs-height" required>
            </div>
        `;
    container.appendChild(div);
  }

  // Condition equations
  const r = numObservations - numParameters;
  const conditionCard = document.getElementById("condition-equations-card");
  const equationsContainer = document.getElementById(
    "condition-equations-inputs"
  );
  equationsContainer.innerHTML = "";

  if (r > 0) {
    conditionCard.style.display = "block";
    for (let i = 1; i <= r; i++) {
      const div = document.createElement("div");
      div.className = "condition-equation-input";
      div.innerHTML = `
                <label class="form-label">Condition Equation #${i}:</label>
                <input type="text" class="form-control condition-equation" 
                       placeholder="e.g., h1 + h2 = B2 - B1" required>
                <small class="text-muted">Use format like h1 + h2 = B2 - B1 or h1 - h2 - h3 = 0</small>
            `;
      equationsContainer.appendChild(div);
    }
  } else {
    conditionCard.style.display = "none";
  }
}

function computeAdjustment() {
  // Validate benchmark inputs
  const benchmarkInputs = document.querySelectorAll(".benchmark-elevation");
  const benchmarks = Array.from(benchmarkInputs)
    .map((input) => parseFloat(input.value.trim()))
    .filter((val) => !isNaN(val));

  if (benchmarks.length === 0) {
    alert("Please enter all benchmark elevations");
    return;
  }

  // Validate observation inputs
  const obsNos = Array.from(document.querySelectorAll(".obs-no")).map((input) =>
    parseInt(input.value)
  );
  const distances = Array.from(document.querySelectorAll(".obs-distance")).map(
    (input) => parseFloat(input.value)
  );
  const hObs = Array.from(document.querySelectorAll(".obs-height")).map(
    (input) => parseFloat(input.value)
  );

  if (
    obsNos.some((v) => isNaN(v)) ||
    distances.some((v) => isNaN(v)) ||
    hObs.some((v) => isNaN(v))
  ) {
    alert("Please complete all observation fields");
    return;
  }

  // Create observations array
  const observations = [];
  for (let i = 0; i < obsNos.length; i++) {
    observations.push({
      Obs_No: obsNos[i],
      Distances_Km: distances[i],
      h_obs: hObs[i],
    });
  }

  // Validate condition equations if needed
  const formulas = [];
  const conditionInputs = document.querySelectorAll(".condition-equation");
  if (conditionInputs.length > 0) {
    conditionInputs.forEach((input) => {
      if (input.value.trim()) {
        formulas.push(input.value.trim());
      }
    });

    if (formulas.length !== conditionInputs.length) {
      alert("Please complete all condition equations");
      return;
    }
  }

  // Prepare data for API
  const data = {
    benchmarks: benchmarks,
    observations: observations,
    num_parameters: parseInt(document.getElementById("num-parameters").value),
    formulas: formulas,
  };

  // Show loading state
  const computeBtn = document.getElementById("compute-btn");
  computeBtn.disabled = true;
  computeBtn.textContent = "Computing...";

  // Call API
  fetch("/api/adjustment", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      displayResults(data);
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Computation failed: " + error.message);
    })
    .finally(() => {
      computeBtn.disabled = false;
      computeBtn.textContent = "Compute Adjustment";
    });
}
g4
function displayResults(data) {
  const container = document.getElementById("results-output");
  container.innerHTML = "";

  // Create a formatted HTML display of the results
  let resultsHTML = `
    <h3 style="text-align: center;"><strong>Basic Information</strong></h3>
    <p>Number of Observations: ${data.num_observations}</p>
    <p>Number of Parameters: ${data.num_parameters}</p>
    <p>Number of Condition Equations: ${data.num_condition_equations}</p>
    
    <h3 style="text-align: center;"><strong>Benchmark Data</strong></h3>
    <p style="text-align: center; font-style: italic;">Benchmark data represents fixed reference points with established elevations.<br>
    These points serve as the foundation for adjustment calculations.<br>
    All new elevations are determined relative to these known benchmark values.</p>
    <table class="table table-bordered">
      <thead>
        <tr>
          <th>Point</th>
          <th>Elevation (m)</th>
        </tr>
      </thead>
      <tbody>
        ${data.benchmark_data
          .map(
            (bench) => `
          <tr>
            <td>${bench.Point}</td>
            <td>${bench.Elevation}</td>
          </tr>
        `
          )
          .join("")}
      </tbody>
    </table>
    
    <h3 style="text-align: center;"><strong>Observation Data</strong></h3>
    <p style="text-align: center; font-style: italic;">Observation data consists of field measurements collected between survey points in a network.
    These measurements include height differences and distances that contain random errors requiring adjustment.</p>
    <table class="table table-bordered">
      <thead>
        <tr>
          <th>Observation #</th>
          <th>Distance (Km)</th>
          <th>Height Difference (m)</th>
        </tr>
      </thead>
      <tbody>
        ${data.observation_data
          .map(
            (obs) => `
          <tr>
            <td>${obs.Obs_No}</td>
            <td>${obs.Distances_Km}</td>
            <td>${obs.h_obs}</td>
          </tr>
        `
          )
          .join("")}
      </tbody>
    </table>
  `;

  // Add condition equation results if available
  if (data.modified_equations) {
    resultsHTML += `
      <h3 style="text-align: center;"><strong>Modified Equations</strong></h3>
      <p style="text-align: center; font-style: italic;">Modified equations convert original condition formulas by adding correction terms (V variables) to height values.
The function transforms expressions like "h1+h2=h3" into "V1+h1+V2+h2=V3+h3" through regex parsing.
This systematic transformation prepares the equations for coefficient extraction. </p>
      <ul>
        ${data.modified_equations.map((eq) => `<li>${eq}</li>`).join("")}
      </ul>
    `;
  }

  // Add distance matrix if available
  if (data.distance_matrix) {
    resultsHTML += `
      <h3 style="text-align: center;"><strong>Distance Matrix</strong></h3>
      <p style="text-align: center; font-style: italic;">Distance matrix is a diagonal matrix containing surveyed distances (in Km) for each observation.
It serves as a weight matrix in adjustment computations, with larger distances receiving less weight.<br>
This matrix is used to scale coefficients in condition equations, reflecting measurement precision relative to observation length. </p>
      <div class="table-responsive">
        <table class="table table-bordered">
          <tbody>
            ${data.distance_matrix
              .map(
                (row) => `
              <tr>
                ${row.map((val) => `<td>${val.toFixed(3)}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Add coefficient table if available
  if (data.coefficient_table && data.coefficient_table.length > 0) {
    const numObs = data.observation_data.length;

    resultsHTML += `
      <h3 style="text-align: center;"><strong>Coefficient Table</strong></h3>
      <p style="text-align: center; font-style: italic;">the coefficient table is a matrix that represents the relationship between each condition equation 
      and the correction parameters (V1, V2, etc.) in the geodetic adjustment process. It consists of rows
       for each condition equation and columns for each observation, with elements containing values (+1, -1, or 0) that
        indicate how each observation contributes to satisfying the geometric constraints expressed by the condition equations.</p>
      <div class="table-responsive">
        <table class="table table-bordered">
          <thead>
            <tr>
              <th>Equation</th>
              ${Array.from(
                { length: numObs },
                (_, i) => `<th>V${i + 1}</th>`
              ).join("")}
            </tr>
          </thead>
          <tbody>
            ${data.coefficient_table
              .map(
                (row) => `
              <tr>
                ${row.map((val) => `<td>${val}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Add weighted coefficient table if available
  if (data.result_table && data.result_table.length > 0) {
    const numObs = data.observation_data.length;

    resultsHTML += `
      <h3 style="text-align: center;"><strong>Weighted Coefficient Table (Coefficient × Distance)</strong></h3>
      <p style="text-align: center; font-style: italic;">The Weighted Coefficient Table is created by multiplying each coefficient in the coefficient table by the 
      corresponding distance value from the distance matrix. This weighting reflects the influence of observation
       distances on the adjustment process, as longer distances typically have greater uncertainty.</p>
      <div class="table-responsive">
        <table class="table table-bordered">
          <thead>
            <tr>
              <th>Equation</th>
              ${Array.from(
                { length: numObs },
                (_, i) => `<th>V${i + 1}</th>`
              ).join("")}
              <th>w</th>
            </tr>
          </thead>
          <tbody>
            ${data.result_table
              .map(
                (row) => `
              <tr>
                ${row.map((val) => `<td>${val}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Add transposed matrix if available
  if (data.transposed_table && data.transposed_table.length > 0) {
    resultsHTML += `
      <h3 style="text-align: center;"><strong>Transposed Matrix</strong></h3>
      <p style="text-align: center; font-style: italic;">The Transposed Matrix is created by switching the rows and columns of the coefficient table, 
      transforming equation-based rows into parameter-based rows. In this geodetic adjustment process, 
      it converts the n×m coefficient matrix to an m×n matrix where m is the number of parameters and n 
      is the number of equations. This transposition is a necessary step for subsequent matrix operations. </p>
      <div class="table-responsive">
        <table class="table table-bordered">
          <thead>
            <tr>
              ${data.transposed_table[0]
                .map((val) => `<th>${val}</th>`)
                .join("")}
            </tr>
          </thead>
          <tbody>
            ${data.transposed_table
              .slice(1)
              .map(
                (row) => `
              <tr>
                ${row.map((val) => `<td>${val}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Add product table if available
  if (data.product_table && data.product_table.length > 0) {
    const numEquations = data.product_table.length;

    resultsHTML += `
      <h3 style="text-align: center;"><strong>Product Table (Result Table × Transpose)</strong></h3>
      <p style="text-align: center; font-style: italic;">The Product Table is the result of multiplying the Weighted Coefficient Table with its transpose,
       creating a square matrix that represents the normal equations in the adjustment process. </p>
      <div class="table-responsive">
        <table class="table table-bordered">
          <thead>
            <tr>
              <th>Equation</th>
              ${Array.from(
                { length: numEquations },
                (_, i) => `<th>Eqn ${i + 1}</th>`
              ).join("")}
            </tr>
          </thead>
          <tbody>
            ${data.product_table
              .map(
                (row) => `
              <tr>
                ${row.map((val) => `<td>${val}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Add inverse table if available
  if (data.inverse_table && data.inverse_table.length > 0) {
    const numEquations = data.inverse_table.length;

    resultsHTML += `
      <h3 style="text-align: center;"><strong>Inverse of Product Table</strong></h3>
      <p style="text-align: center; font-style: italic;">The Inverse of Product Table is the matrix inverse of the Product Table (normal equations matrix),
       which is a crucial component for solving the condition equation adjustment system. This inverse matrix, 
       when multiplied by the w values (condition equation constants), yields the Lagrange multipliers (K values) 
       that are needed to determine corrections to observations. The inverse operation effectively solves the system 
       of equations that balances all geometric constraints in the adjustment process. </p>
      <div class="table-responsive">
        <table class="table table-bordered">
          <thead>
            <tr>
              <th>Equation</th>
              ${Array.from(
                { length: numEquations },
                (_, i) => `<th>Eqn ${i + 1}</th>`
              ).join("")}
            </tr>
          </thead>
          <tbody>
            ${data.inverse_table
              .map(
                (row) => `
              <tr>
                ${row.map((val) => `<td>${val}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Add K values if available
  if (data.v_table_from_inverse && data.v_table_from_inverse.length > 0) {
    resultsHTML += `
      <h3 style="text-align: center;"><strong>K Values</strong></h3>
      <p style="text-align: center; font-style: italic;">K values are the Lagrange multipliers calculated by multiplying the negative 
      inverse of the Product Table with the w values (condition equation constants). 
      These K values are essential coefficients in the condition equation adjustment 
      method that help determine the optimal corrections to observations. </p>
      <table class="table table-bordered">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          ${data.v_table_from_inverse
            .map(
              (row) => `
            <tr>
              <td>${row[0]}</td>
              <td>${row[1]}</td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    `;
  }

  // Add residuals if available
  if (data.residual && data.residual.length > 0) {
    resultsHTML += `
      <h3 style="text-align: center;"><strong>Residuals</strong></h3>
      <p style="text-align: center; font-style: italic;">Residuals represent the corrections applied to observed height differences 
      (h_obs) to obtain the most consistent set of measurements. They're calculated by 
      multiplying the negative weight matrix (-P) with the transpose of the coefficient matrix 
      and parameter estimates (K-values). These residuals help minimize errors in the height measurements 
      based on the mathematical constraints defined in the condition equations. </p>
      <table class="table table-bordered">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          ${data.residual
            .map(
              (row) => `
            <tr>
              <td>${row[0]}</td>
              <td>${row[1]}</td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    `;
  }
  // Add adjusted heights if available
  if (data.adjusted_heights) {
    resultsHTML += `
      <h3 style="text-align: center;"><strong>Adjusted Heights</strong></h3>
      <p style="text-align: center; font-style: italic;">Adjusted heights are the final corrected height values obtained by adding 
      the calculated residuals to the originally observed height differences (h_obs). 
      They represent the most consistent set of height measurements after accounting 
      for mathematical constraints and minimizing errors. The adjusted heights are 
      computed in the calculate_adjusted_heights function as: adjusted_height = h_obs + residual_value. </p>
      <table class="table table-bordered">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          ${data.adjusted_heights
            .map(
              (adj) => `
            <tr>
              <td>${adj[0]}</td>
              <td>${adj[1]}</td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    `;
  }

  container.innerHTML = resultsHTML;

  // Scroll to results
  container.scrollIntoView({ behavior: "smooth" });
}