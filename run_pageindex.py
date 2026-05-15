#导入依赖
import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

#程序入口
if __name__ == "__main__":
    # Set up argument parser
    #解析命令行参数  定义你可以在命令行里传哪些参数      命令行参数最后都变成了 args.xxx
    #parser.add_argument(...) 是在“声明这个脚本支持哪些命令行参数”。
    #parser是argparse.ArgumentParser 类的实例对象，负责定义、解析命令行参数
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')
    #type=int 表示解析时会转成整数   args.toc_check_pages == 20   是整数 20，不是字符串 "20"
    #如果你没传某个参数，它就用 default。这里大部分默认值是 None
    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')

 
    #                   
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args() #真正开始解析命令行，并把结果放进 args
    
    #args 是 argparse 把命令行里传进来的参数解析后生成的一个对象。
    # 它不是普通 dict，而是 argparse.Namespace 对象。



    # Validate that exactly one file type is specified

    #要求 PDF 和 Markdown 只能选一个  不能两个都传  这个脚本有两个分支
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    if args.pdf_path:
        # Validate PDF file
        #校验文件是不是 PDF
        #lower是为了兼容大小写的后缀
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        #校验文件是否存在
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Process PDF file
        #组装用户配置、这一步是把命令行参数整理成 PageIndex 内部使用的配置字典。
        user_opt = {
            'model': args.model,
            'toc_check_page_num': args.toc_check_pages,
            'max_page_num_each_node': args.max_pages_per_node,
            'max_token_num_each_node': args.max_tokens_per_node,
            'if_add_node_id': args.if_add_node_id,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
        }
        #合并命令行参数和config.yaml
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        # Process the PDF
        #传入的两个参数分别是PDF 文件路径和配置对象opt
        #这是 PDF 主流程真正开始的地方  这就是run_pageindex.py 本身不负责解析 PDF 的细节，
        # 只负责把PDF路径和配置合并  再交给page_index_main(...)
        #等它返回一个结构化结果  输出的是一个树状JSON
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results
        #提取纯 PDF 文件名  从(纯文件名，扩展名) 的元组提取第一个元素也就是不带扩展名的纯文本名
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
        output_dir = './results/indexes'#定义输出目录
        output_file = f'{output_dir}/{pdf_name}_structure.json'  #拼接完整的输出文件路径
        os.makedirs(output_dir, exist_ok=True)#创建输出目录
        
        #写入json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_dir = './results/indexes'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
