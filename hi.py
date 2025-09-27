 # Plot the response curve
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    v = np.linspace(0, 1, 512)
    resp = (v**n_naka) / (v**n_naka + (sigma_g**n_naka + 1e-12))
    ax.plot(v, resp, lw=2)
    ax.set_title("Naka–Rushton Response Curve")
    ax.set_xlabel("Input luminance V")
    ax.set_ylabel("Adapted luminance V'")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Before/After luminance heatmaps and histograms
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        imshow_heat(ax, V_before, "Luminance V (before)", cmap="magma")
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        hist_plot(ax, V_before, "Histogram V (before)")
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        imshow_heat(ax, V_after, "Luminance V' (after)", cmap="magma")
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        hist_plot(ax, V_after, "Histogram V' (after)")
        st.pyplot(fig, use_container_width=True)

# ---------- FUSION ----------
with tab_fuse:
    st.markdown("**Weighted fusion**: "
                r"$I_{out} = I_{enh\_base} + \omega_c \cdot D_c$, where "
                r"$\omega_c = \alpha_c (|D_c| * \mathcal{N}_\sigma)$.")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for c, name in enumerate(["R", "G", "B"]):
            imshow_heat(axes[0,c], np.abs(detail[..., c]), f"|Detail| channel {name}", cmap="magma")
        for c, name in enumerate(["R", "G", "B"]):
            imshow_heat(axes[1,c], weights[..., c], f"Weight ω_{name}", cmap="viridis")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.image(to_uint8(enh_base), caption="Enhanced Base (pre-fusion)", use_container_width=True)
        st.image(to_uint8(out), caption="Final Output (post-fusion)", use_container_width=True)

# ---------- LINE PROFILES ----------
with tab_profiles:
    st.markdown("**Contrast along a line**: compares grayscale profiles before/after to illustrate vessel enhancement.")
    # Choose a central horizontal line across the optic region
    H, W = rgb01.shape[:2]
    r0 = r1 = H // 2
    c0, c1 = W // 6, 5 * W // 6

    # Use green channel (clinically informative) for profiles
    g_before = rgb01[..., 1]
    g_after  = out[..., 1]
    prof_before, coords = profile_line(g_before, r0, r1, c0, c1, num=800)
    prof_after, _ = profile_line(g_after,  r0, r1, c0, c1, num=800)

    # Show the sampling line
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(rgb01)
    ax[0].plot([c0, c1], [r0, r1], lw=2)
    ax[0].set_title("Sampling line on Original")
    ax[0].axis("off")
    ax[1].imshow(out)
    ax[1].plot([c0, c1], [r0, r1], lw=2)
    ax[1].set_title("Sampling line on Enhanced")
    ax[1].axis("off")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Plot profiles
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(np.linspace(0, 1, prof_before.size), prof_before, label="Before", lw=1.8)
    ax.plot(np.linspace(0, 1, prof_after.size),  prof_after,  label="After", lw=1.8)
    ax.set_title("Intensity Profile Along the Line (G channel)")
    ax.set_xlabel("Normalized distance")
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ==================================
# =========== Download =============
# ==================================
buf = io.BytesIO()
Image.fromarray(to_uint8(out)).save(buf, format="PNG")
st.download_button("⬇️ Download enhanced PNG", data=buf.getvalue(),
                   file_name="fundus_enhanced.png", mime="image/png")

# ==================================
# ======= Deployment notes =========
# ==================================
st.caption("Tip: For servers or Streamlit Cloud, depend on `opencv-python-headless` to avoid GUI deps.")
