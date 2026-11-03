"""Single-command entry point."""
import argparse
from .pipeline import Agent

def main():
    parser=argparse.ArgumentParser(prog='fade-content-agent',description='Generate a validated bilingual subtitle learning package with Gemini.')
    parser.add_argument('subtitle_file',help='SRT/VTT path, YouTube URL, or 11-character video ID'); parser.add_argument('--source-language'); parser.add_argument('--target-language',required=True); parser.add_argument('--video-id'); parser.add_argument('--language',help='Preferred YouTube subtitle language; omitted uses TranscriptAPI default selection'); parser.add_argument('--title'); parser.add_argument('--context'); parser.add_argument('--glossary',help='JSON glossary file'); parser.add_argument('--output-dir',default='fade-output'); parser.add_argument('--batch-size',type=int,default=50); parser.add_argument('--resume',metavar='RUN_ID',help='Resume a previously interrupted run using its saved intermediate results')
    args=parser.parse_args(); glossary=None
    if args.glossary:
        import json
        with open(args.glossary,encoding='utf-8') as f: glossary=json.load(f)
    path=Agent(args.output_dir,batch_size=args.batch_size).run(args.subtitle_file,args.source_language,args.target_language,args.video_id,title=args.title,context=args.context,glossary=glossary,run_id=args.resume,language=args.language)
    print(f'Validated package created: {path}')

if __name__=='__main__': main()
