# Hourai's Template

Hourai's Template 是哈尔滨工业大学蓬莱人形队伍在 2022~2024 赛季参加中国大学生程序设计竞赛（CCPC）和国际大学生程序设计竞赛（ICPC）期间所用模板的归档整理。包含了大量算法竞赛常见算法的代码实现。

该项目的工作方式是，利用自动构建脚本将源代码以及注释生成为 Markdown 文件，再利用 pandoc + XeLaTeX 生成 pdf 文件。Markdown 文件同时会被应用在网页展示上。然而，由于不同环境行为不同，网页构建效果可能与 pdf 不同，在此优先保证 pdf 格式，网页保证基本的查阅功能以及代码的渲染。

此外，由于 ICPC World Final 规则只允许携带 25 页模板（不含封面），因而我们对代码进行了一定程度的压缩，选取其中比较重要的一些模板整合成了精简版本。

本项目受 [OI Wiki](https://oi-wiki.org/) 的启发，参考了相关实现。

## 算法选择

算法知识点主要参考了 NOI 大纲以及编写者的 OI & ACM 经历。

基于赛事考察范围，本模板选择的算法有如下考量：代码长度较短、不需要依靠板子就可默写出来的算法没有选择；超出考纲范围，偏门冷门奇形怪状的代码没有选择。

考虑到赛时有限的机时，大多数代码没有经过严格完整的封装，对于较长的代码仅使用命名空间进行组织，也没有使用过多的宏定义进行包装，方便能够在赛时快速抄写。

## 可携带版本

通过自动化脚本构建出的完整模板可以在以下链接处查看。

- 基于全部内容形成的完整版本：
    - 网页：<https://hourai.nanani-fan.club/offline/print>；
    - 文档：<https://hourai.nanani-fan.club/offline/print.pdf>。
- 基于精简内容形成的 World Final 特供版本：
    - 网页：<https://hourai.nanani-fan.club/offline/print-wf>；
    - 文档：<https://hourai.nanani-fan.club/offline/print-wf.pdf>。

不保证网页版本的文档结构等行为正确，仅供参考。如需检索可以使用 Mkdocs 提供的网页搜索功能。

## 关于贡献

因为是自用模板，所以暂时没有接受贡献的计划，可能会随着比赛参与过程进行微调。

## 关于使用

本模板里的代码全部由编写者亲自完成，基于 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en) 协议提供。任何队伍可自由地使用该模板参与比赛。