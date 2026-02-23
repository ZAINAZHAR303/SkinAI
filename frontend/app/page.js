"use client";

import { useState, useRef } from "react";

export default function Home() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      console.log("📤 Sending request to backend...");

      let res;
      try {
        res = await fetch("http://localhost:8000/predict", {
          method: "POST",
          body: formData,
        });
      } catch (networkErr) {
        console.error("🔴 Network error:", networkErr);
        throw new Error(
          "Cannot connect to the backend (ERR_CONNECTION_REFUSED). " +
          "Make sure the FastAPI server is running: uvicorn main:app --reload"
        );
      }

      console.log("📥 Response status:", res.status, res.statusText);

      if (!res.ok) {
        let detail = "";
        try {
          const errBody = await res.json();
          detail = errBody.detail || JSON.stringify(errBody);
        } catch {
          detail = await res.text();
        }
        console.error("🔴 Server error:", detail);
        throw new Error(`Server error ${res.status}: ${detail}`);
      }

      const data = await res.json();
      console.log("✅ Prediction result:", data);
      setResult(data);
    } catch (err) {
      console.error("❌ handlePredict error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <main className="min-h-screen bg-slate-50 font-sans">
      {/* Navbar */}
      <nav className="bg-white border-b shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="text-xl font-bold text-blue-600">SkinAI</span>
          <span className="text-sm text-slate-500">AI-Powered Skin Disease Detection</span>
        </div>
      </nav>

      {/* Hero */}
      <section className="bg-gradient-to-br from-blue-600 to-indigo-700 text-white py-20 px-6 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
          Skin Disease Detection
        </h1>
        <p className="text-lg md:text-xl text-blue-100 max-w-2xl mx-auto mb-8">
          Upload a photo of a skin lesion and get an instant AI-powered diagnosis
          using our Convolutional Neural Network model.
        </p>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="bg-white text-blue-600 font-semibold px-8 py-3 rounded-full shadow-lg hover:bg-blue-50 transition"
        >
          Upload Image
        </button>
      </section>

      {/* Upload & Predict */}
      <section className="max-w-2xl mx-auto px-6 py-16">
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-center mb-6 text-slate-800">
            Analyze a Skin Lesion
          </h2>

          {/* Dropzone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition mb-6"
          >
            {preview ? (
              <img
                src={preview}
                alt="Preview"
                className="mx-auto max-h-64 rounded-lg object-contain"
              />
            ) : (
              <>
                <div className="text-5xl mb-3">🖼️</div>
                <p className="text-slate-500">
                  Click to select an image, or drag &amp; drop
                </p>
                <p className="text-xs text-slate-400 mt-1">JPG, PNG, WEBP</p>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handlePredict}
              disabled={!image || loading}
              className="flex-1 bg-blue-600 text-white font-semibold py-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {loading ? "Analyzing..." : "Predict"}
            </button>
            {(image || result) && (
              <button
                onClick={handleReset}
                className="px-5 py-3 rounded-xl border border-slate-300 text-slate-600 hover:bg-slate-100 transition"
              >
                Reset
              </button>
            )}
          </div>

          {/* Result */}
          {result && (
            <div className="mt-6 bg-green-50 border border-green-200 rounded-xl p-5 text-center">
              <p className="text-sm text-green-600 font-medium uppercase tracking-wide mb-1">
                Prediction
              </p>
              <p className="text-2xl font-bold text-green-800">{result.class}</p>
              <p className="text-slate-500 mt-1">
                Confidence:{" "}
                <span className="font-semibold text-slate-700">
                  {result.confidence}%
                </span>
              </p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4 text-center text-red-600 text-sm">
              {error}
            </div>
          )}

          <p className="text-xs text-slate-400 text-center mt-4">
            ⚠️ For informational purposes only. Not a substitute for medical advice.
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-slate-100 py-16">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-10 text-slate-800">
            How It Works
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: "📤",
                title: "Upload Image",
                desc: "Select a clear photo of the skin condition from your device.",
              },
              {
                icon: "🧠",
                title: "AI Processing",
                desc: "Our CNN model processes the image and extracts medical features.",
              },
              {
                icon: "✅",
                title: "Instant Result",
                desc: "The system returns the predicted skin disease with confidence.",
              },
            ].map((item, i) => (
              <div
                key={i}
                className="bg-white rounded-2xl shadow-lg p-6 text-center"
              >
                <div className="text-4xl mb-3">{item.icon}</div>
                <h3 className="font-semibold text-lg mb-2 text-slate-800">
                  {item.title}
                </h3>
                <p className="text-slate-600 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Detectable Conditions */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-center mb-10 text-slate-800">
          Detectable Conditions
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          {[
            "Melanoma",
            "Nevus",
            "Basal Cell Carcinoma",
            "Actinic Keratosis",
            "Benign Keratosis",
            "Dermatofibroma",
            "Vascular Lesion",
          ].map((condition, i) => (
            <div
              key={i}
              className="bg-blue-50 border border-blue-100 rounded-xl p-4"
            >
              <p className="font-medium text-blue-800 text-sm">{condition}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Key Benefits */}
      <section className="bg-white py-16 border-t">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-10 text-slate-800">
            Key Benefits
          </h2>

          <div className="grid md:grid-cols-4 gap-6 text-center">
            {[
              { icon: "⚡", text: "Fast preliminary screening" },
              { icon: "🤖", text: "AI-assisted diagnosis" },
              { icon: "🌐", text: "Accessible anywhere" },
              { icon: "🔬", text: "Supports medical research" },
            ].map((b, i) => (
              <div key={i} className="p-4">
                <div className="text-3xl mb-2">{b.icon}</div>
                <p className="font-medium text-slate-700">{b.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-center mb-10 text-slate-800">
          Use Cases
        </h2>

        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: "🏥",
              title: "Clinical Decision Support",
              desc: "Assist dermatologists in preliminary screening.",
            },
            {
              icon: "📱",
              title: "Telemedicine",
              desc: "Enable remote skin condition assessment.",
            },
            {
              icon: "📊",
              title: "Medical Research",
              desc: "Support AI research in computer vision for healthcare.",
            },
          ].map((item, i) => (
            <div
              key={i}
              className="bg-white rounded-2xl shadow-lg p-6 text-center"
            >
              <div className="text-4xl mb-3">{item.icon}</div>
              <h3 className="font-semibold text-lg mb-2 text-slate-800">
                {item.title}
              </h3>
              <p className="text-slate-600 text-sm">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-800 text-center py-8 text-slate-400 text-sm">
        Built by Zain Azhar &bull; AI for Healthcare
      </footer>
    </main>
  );
}