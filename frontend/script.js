const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const patientData = {

        age: Number(document.getElementById("age").value),
        bp: Number(document.getElementById("bp").value),
        sg: Number(document.getElementById("sg").value),
        al: Number(document.getElementById("al").value),
        su: Number(document.getElementById("su").value),
        bgr: Number(document.getElementById("bgr").value),
        bu: Number(document.getElementById("bu").value),
        sc: Number(document.getElementById("sc").value),
        sod: Number(document.getElementById("sod").value),
        pot: Number(document.getElementById("pot").value),
        hemo: Number(document.getElementById("hemo").value),
        pcv: Number(document.getElementById("pcv").value),
        wbcc: Number(document.getElementById("wbcc").value),
        rbcc: Number(document.getElementById("rbcc").value),

        rbc: document.getElementById("rbc").value,
        pc: document.getElementById("pc").value,
        pcc: document.getElementById("pcc").value,
        ba: document.getElementById("ba").value,
        htn: document.getElementById("htn").value,
        dm: document.getElementById("dm").value,
        cad: document.getElementById("cad").value,
        appet: document.getElementById("appet").value,
        pe: document.getElementById("pe").value,
        ane: document.getElementById("ane").value
    };


    try {

        const response = await fetch(
           "https://ckd-ml-prediction-system.onrender.com/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(patientData)
            }
        );


        if (!response.ok) {
            throw new Error("Prediction request failed");
        }


        const result = await response.json();


        document.getElementById("result")
            .classList.remove("hidden");


        document.getElementById("prediction")
            .textContent = result.prediction;


        document.getElementById("probability")
            .textContent =
            (result.probability * 100).toFixed(2) + "%";


    } catch (error) {

        console.error(error);

        alert(
            "Unable to connect to the prediction server."
        );
    }

});