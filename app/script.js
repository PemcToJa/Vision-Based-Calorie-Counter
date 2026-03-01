document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    const inputRgb = document.getElementById('input-rgb');
    const btnAnalyze = document.getElementById('btn-analyze');
    const rgbPreview = document.getElementById('prev-rgb');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');

    inputRgb.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                rgbPreview.src = ev.target.result;
                rgbPreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
        validate();
    });

    function validate() {
        const hasRgb = inputRgb.files.length > 0;
        btnAnalyze.disabled = !hasRgb;
    }

    btnAnalyze.addEventListener('click', async () => {
        if (inputRgb.files.length === 0) return;

        const formData = new FormData();
        formData.append('model_type', 'fusion');
        formData.append('rgb_image', inputRgb.files[0]);

        setUIState(true);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                updateResults(data);
            } else {
                alert("Error: " + (data.detail || "An unexpected problem occurred."));
            }
        } catch (error) {
            console.error("Conection error:", error);
            alert("Server error. Make sure the Python application is running.");
        } finally {
            setUIState(false);
        }
    });

    function setUIState(loading) {
        btnAnalyze.disabled = loading;
        loader.classList.toggle('hidden', !loading);
        btnText.innerText = loading ? 'Analyzing...' : 'Analyze the meal';

        if (loading) {
            btnAnalyze.style.opacity = "0.7";
            btnAnalyze.style.cursor = "wait";
        } else {
            btnAnalyze.style.opacity = "1";
            btnAnalyze.style.cursor = "pointer";
        }
    }

    function updateResults(data) {
        document.getElementById('empty-state').classList.add('hidden');
        document.getElementById('stats-grid').classList.remove('hidden');

        document.getElementById('res-kcal').innerText = Math.round(data.calories);
        document.getElementById('res-prot').innerText = data.protein;
        document.getElementById('res-carb').innerText = data.carbs;
        document.getElementById('res-fat').innerText = data.fat;

        const resType = document.getElementById('res-type');
        if (resType) {
            resType.innerText = `${data.product_type} (~${data.mass}g)`;
        }

        updateDonutChart(data.protein, data.carbs, data.fat);
    }

    function updateDonutChart(prot, carb, fat) {
        const total = prot + carb + fat;
        const dFat = document.getElementById('d-fat');
        const dCarb = document.getElementById('d-carb');
        const dProt = document.getElementById('d-prot');

        if (total > 0) {
            const pFat = (fat / total) * 100;
            const pCarb = (carb / total) * 100;
            const pProt = (prot / total) * 100;

            dFat.style.strokeDasharray = `${pFat} 100`;

            dCarb.style.strokeDasharray = `${pCarb} 100`;
            dCarb.style.strokeDashoffset = `-${pFat}`;

            dProt.style.strokeDasharray = `${pProt} 100`;
            dProt.style.strokeDashoffset = `-${pFat + pCarb}`;
        }
    }
});