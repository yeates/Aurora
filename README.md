# Aurora: Unified Video Editing with a Tool-Using Agent

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2605.18748) [![Website](https://img.shields.io/badge/Website-Project-6535a0)](https://yeates.github.io/Aurora-Page) [![Code](https://img.shields.io/badge/Code-Late_May_2026-A55D35.svg)](#code-release)

This repository will host the official implementation of **Aurora**, an agentic video editing framework that pairs a tool-augmented vision-language model (VLM) agent with a unified video diffusion transformer. The VLM agent rewrites a raw user request into a typed edit plan (instruction, task label, image-search query, mask phrase) and dispatches it to the video DiT, resolving textual and visual underspecification before generation.

## Features

* 🎬 **Unified video editing** - replacement, removal, style transfer, and reference-driven insertion under one set of weights
* 🤖 **Tool-using VLM agent** - rewrites a raw user request into a four-field edit plan
* 🔍 **Resolves underspecification** - fills in missing reference images via web image search and missing masks via grounded segmentation
* 📊 **AgentEdit-Bench** - evaluates agent-enhanced video editing under textual and visual underspecification

## TODO

- [] The code is being prepared for release. ETA: late May 2026.
