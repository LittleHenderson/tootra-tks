use crate::ast::*;
use crate::lexer::{Lexer, LexerError, Token, TokenKind};
use crate::span::Span;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Option<Span>,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl From<LexerError> for ParseError {
    fn from(value: LexerError) -> Self {
        ParseError::new(value.message, None)
    }
}

pub fn parse_program(source: &str) -> Result<Program, ParseError> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    stop_at_pipe: bool,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            stop_at_pipe: false,
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut decls = Vec::new();
        while self.is_top_decl_start() {
            decls.push(self.parse_top_decl()?);
            self.match_kind(&TokenKind::Semicolon);
        }
        let entry = if self.check(&TokenKind::Eof) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(&TokenKind::Eof, "expected end of input")?;
        Ok(Program { decls, entry })
    }

    fn is_top_decl_start(&self) -> bool {
        matches!(
            self.peek_kind(),
            TokenKind::Let
                | TokenKind::TypeKw
                | TokenKind::Effect
                | TokenKind::Handler
                | TokenKind::Module
                | TokenKind::External
        )
    }

    fn parse_top_decl(&mut self) -> Result<TopDecl, ParseError> {
        match self.peek_kind() {
            TokenKind::Let => self.parse_let_decl(),
            TokenKind::TypeKw => self.parse_type_decl(),
            TokenKind::Effect => self.parse_effect_decl(),
            TokenKind::Handler => self.parse_handler_decl(),
            TokenKind::Module => self.parse_module_decl(),
            TokenKind::External => self.parse_extern_decl(),
            _ => Err(self.error_here("expected top-level declaration")),
        }
    }

    fn parse_let_decl(&mut self) -> Result<TopDecl, ParseError> {
        let start = self.expect(&TokenKind::Let, "expected 'let'")?;
        let name = self.expect_ident("expected identifier after 'let'")?;
        let scheme = self.parse_optional_type_scheme()?;
        self.expect(&TokenKind::Equals, "expected '=' after let binding")?;
        let value = self.parse_expr()?;
        let span = span_join(start.span, expr_span(&value));
        Ok(TopDecl::LetDecl {
            span,
            name,
            scheme,
            value,
        })
    }

    fn parse_optional_type_scheme(&mut self) -> Result<Option<TypeScheme>, ParseError> {
        let mut vars = Vec::new();
        if self.match_kind(&TokenKind::LBracket) {
            if !self.check(&TokenKind::RBracket) {
                loop {
                    vars.push(self.expect_ident("expected type variable")?);
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RBracket, "expected ']' after type variables")?;
        }
        if self.match_kind(&TokenKind::Colon) {
            let ty = self.parse_type()?;
            return Ok(Some(TypeScheme { vars, ty }));
        }
        if !vars.is_empty() {
            return Err(self.error_here("expected ':' after type variables"));
        }
        Ok(None)
    }

    fn parse_type_decl(&mut self) -> Result<TopDecl, ParseError> {
        let start = self.expect(&TokenKind::TypeKw, "expected 'type'")?;
        let name = self.expect_ident("expected type name")?;
        let mut params = Vec::new();
        if self.match_kind(&TokenKind::LBracket) {
            if !self.check(&TokenKind::RBracket) {
                loop {
                    params.push(self.expect_ident("expected type parameter")?);
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RBracket, "expected ']' after type parameters")?;
        }
        self.expect(&TokenKind::Equals, "expected '=' in type declaration")?;
        let body = self.parse_type()?;
        let span = span_join(start.span, self.previous_span());
        Ok(TopDecl::TypeDecl {
            span,
            name,
            params,
            body,
        })
    }
    fn parse_effect_decl(&mut self) -> Result<TopDecl, ParseError> {
        let start = self.expect(&TokenKind::Effect, "expected 'effect'")?;
        let name = self.expect_ident("expected effect name")?;
        let mut params = Vec::new();
        if self.match_kind(&TokenKind::LBracket) {
            if !self.check(&TokenKind::RBracket) {
                loop {
                    params.push(self.expect_ident("expected effect type parameter")?);
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RBracket, "expected ']' after effect parameters")?;
        }
        self.expect(&TokenKind::LBrace, "expected '{' to start effect body")?;
        let mut ops = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            ops.push(self.parse_op_sig()?);
            self.match_kind(&TokenKind::Semicolon);
        }
        let end = self.expect(&TokenKind::RBrace, "expected '}' after effect body")?;
        let span = span_join(start.span, end.span);
        Ok(TopDecl::EffectDecl {
            span,
            name,
            params,
            ops,
        })
    }

    fn parse_op_sig(&mut self) -> Result<OpSig, ParseError> {
        let start = self.expect(&TokenKind::Op, "expected 'op'")?;
        let name = self.expect_ident("expected operation name")?;
        let mut param_types = Vec::new();
        if self.match_kind(&TokenKind::LParen) {
            if !self.check(&TokenKind::RParen) {
                loop {
                    self.expect_ident("expected parameter name")?;
                    self.expect(&TokenKind::Colon, "expected ':' in parameter")?;
                    let param_ty = self.parse_type()?;
                    param_types.push(param_ty);
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RParen, "expected ')' after parameters")?;
        }
        self.expect(&TokenKind::Colon, "expected ':' after operation name")?;
        let output = self.parse_type()?;
        let input = build_param_type(param_types);
        let span = span_join(start.span, self.previous_span());
        Ok(OpSig {
            span,
            name,
            input,
            output,
        })
    }

    fn parse_handler_decl(&mut self) -> Result<TopDecl, ParseError> {
        let start = self.expect(&TokenKind::Handler, "expected 'handler'")?;
        let name = self.expect_ident("expected handler name")?;
        let handler_type = if self.match_kind(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let def = self.parse_handler_def_block()?;
        let span = span_join(start.span, def.span);
        Ok(TopDecl::HandlerDecl {
            span,
            name,
            handler_type,
            def,
        })
    }

    fn parse_handler_def_block(&mut self) -> Result<HandlerDef, ParseError> {
        let start = self.expect(&TokenKind::LBrace, "expected '{' to start handler")?;
        let return_clause = self.parse_return_clause()?;
        let mut op_clauses = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            if self.match_kind(&TokenKind::Semicolon) {
                continue;
            }
            op_clauses.push(self.parse_op_clause()?);
            self.match_kind(&TokenKind::Semicolon);
        }
        let end = self.expect(&TokenKind::RBrace, "expected '}' after handler")?;
        let span = span_join(start.span, end.span);
        Ok(HandlerDef {
            span,
            return_clause,
            op_clauses,
        })
    }

    fn parse_return_clause(&mut self) -> Result<(Ident, Expr), ParseError> {
        self.expect(&TokenKind::Return, "expected 'return' clause")?;
        let name = self.expect_ident("expected return binder")?;
        self.expect(&TokenKind::Arrow, "expected '->' in return clause")?;
        let expr = self.parse_expr()?;
        Ok((name, expr))
    }

    fn parse_op_clause(&mut self) -> Result<OpClause, ParseError> {
        let start = if self.match_kind(&TokenKind::Op) {
            self.previous_span()
        } else {
            self.peek().span
        };
        let op = self.expect_ident("expected operation name")?;
        self.expect(&TokenKind::LParen, "expected '(' after op name")?;
        let arg = self.expect_ident("expected operation argument")?;
        self.expect(&TokenKind::RParen, "expected ')' after op argument")?;
        let k = self.expect_ident("expected continuation name")?;
        self.expect(&TokenKind::Arrow, "expected '->' after continuation")?;
        let body = self.parse_expr()?;
        let span = span_join(start, expr_span(&body));
        Ok(OpClause {
            span,
            op,
            arg,
            k,
            body,
        })
    }
    fn parse_module_decl(&mut self) -> Result<TopDecl, ParseError> {
        let start = self.expect(&TokenKind::Module, "expected 'module'")?;
        let path = self.parse_path()?;
        let body = self.parse_module_body()?;
        let span = span_join(start.span, body.span);
        Ok(TopDecl::ModuleDecl(ModuleDecl { span, path, body }))
    }

    fn parse_module_body(&mut self) -> Result<ModuleBody, ParseError> {
        let start = self.expect(&TokenKind::LBrace, "expected '{' to start module")?;
        let exports = if self.check(&TokenKind::Export) {
            Some(self.parse_export_decl()?)
        } else {
            None
        };
        let mut imports = Vec::new();
        while self.check(&TokenKind::Import) || self.check(&TokenKind::From) {
            imports.push(self.parse_import_decl()?);
            self.match_kind(&TokenKind::Semicolon);
        }
        let mut decls = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            if self.is_top_decl_start() {
                decls.push(self.parse_top_decl()?);
                self.match_kind(&TokenKind::Semicolon);
            } else {
                return Err(self.error_here("expected declaration in module body"));
            }
        }
        let end = self.expect(&TokenKind::RBrace, "expected '}' after module")?;
        let span = span_join(start.span, end.span);
        Ok(ModuleBody {
            span,
            exports,
            imports,
            decls,
        })
    }

    fn parse_export_decl(&mut self) -> Result<ExportDecl, ParseError> {
        let start = self.expect(&TokenKind::Export, "expected 'export'")?;
        self.expect(&TokenKind::LBrace, "expected '{' after export")?;
        let mut items = Vec::new();
        if !self.check(&TokenKind::RBrace) {
            loop {
                items.push(self.parse_export_item()?);
                if !self.match_kind(&TokenKind::Comma) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace, "expected '}' after export list")?;
        let span = span_join(start.span, end.span);
        Ok(ExportDecl { span, items })
    }

    fn parse_export_item(&mut self) -> Result<ExportItem, ParseError> {
        if self.match_kind(&TokenKind::TypeKw) {
            let name = self.expect_ident("expected type name in export")?;
            let transparent = self.match_kind(&TokenKind::Equals);
            return Ok(ExportItem::Type { name, transparent });
        }
        if self.match_kind(&TokenKind::Effect) || self.match_kind(&TokenKind::Module) {
            let name = self.expect_ident("expected name in export")?;
            return Ok(ExportItem::Value(name));
        }
        let name = self.expect_ident("expected export item")?;
        Ok(ExportItem::Value(name))
    }

    fn parse_import_decl(&mut self) -> Result<ImportDecl, ParseError> {
        if self.check(&TokenKind::Import) {
            let start = self.expect(&TokenKind::Import, "expected 'import'")?;
            let path = self.parse_path()?;
            if self.match_kind(&TokenKind::As) {
                let alias = self.expect_ident("expected alias after 'as'")?;
                let span = span_join(start.span, self.previous_span());
                return Ok(ImportDecl::Aliased { span, path, alias });
            }
            let span = span_join(start.span, self.previous_span());
            return Ok(ImportDecl::Qualified { span, path });
        }
        let start = self.expect(&TokenKind::From, "expected 'from'")?;
        let path = self.parse_path()?;
        self.expect(&TokenKind::Import, "expected 'import' in from-import")?;
        if self.match_kind(&TokenKind::Times) {
            let span = span_join(start.span, self.previous_span());
            return Ok(ImportDecl::Wildcard { span, path });
        }
        self.expect(&TokenKind::LBrace, "expected '{' for import list")?;
        let mut items = Vec::new();
        if !self.check(&TokenKind::RBrace) {
            loop {
                items.push(self.parse_import_item()?);
                if !self.match_kind(&TokenKind::Comma) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RBrace, "expected '}' after import list")?;
        let span = span_join(start.span, self.previous_span());
        Ok(ImportDecl::Selective { span, path, items })
    }

    fn parse_import_item(&mut self) -> Result<ImportItem, ParseError> {
        if self.match_kind(&TokenKind::TypeKw) || self.match_kind(&TokenKind::Effect) {
            let name = self.expect_ident("expected name in import item")?;
            return Ok(ImportItem { name, alias: None });
        }
        let name = self.expect_ident("expected import item")?;
        let alias = if self.match_kind(&TokenKind::As) {
            Some(self.expect_ident("expected alias name")?)
        } else {
            None
        };
        Ok(ImportItem { name, alias })
    }

    fn parse_path(&mut self) -> Result<Vec<Ident>, ParseError> {
        let mut path = Vec::new();
        path.push(self.expect_ident("expected module path")?);
        while self.match_kind(&TokenKind::Dot) {
            path.push(self.expect_ident("expected module segment")?);
        }
        Ok(path)
    }

    fn parse_extern_decl(&mut self) -> Result<TopDecl, ParseError> {
        let start = self.expect(&TokenKind::External, "expected 'external'")?;
        let convention_name = self.expect_ident("expected calling convention")?;
        let convention = match convention_name.to_ascii_lowercase().as_str() {
            "c" => Convention::C,
            "stdcall" => Convention::StdCall,
            "fastcall" => Convention::FastCall,
            "system" => Convention::System,
            _ => return Err(self.error_here("unknown calling convention")),
        };
        let safety = if self.match_kind(&TokenKind::Safe) {
            Safety::Safe
        } else if self.match_kind(&TokenKind::Unsafe) {
            Safety::Unsafe
        } else {
            Safety::Safe
        };
        self.expect(&TokenKind::FnKw, "expected 'fn' in external declaration")?;
        let name = self.expect_ident("expected external function name")?;
        self.expect(&TokenKind::LParen, "expected '(' in external declaration")?;
        let mut params = Vec::new();
        if !self.check(&TokenKind::RParen) {
            loop {
                let param_name = self.expect_ident("expected parameter name")?;
                self.expect(&TokenKind::Colon, "expected ':' in parameter")?;
                let param_ty = self.parse_type()?;
                params.push(ExternParam {
                    name: param_name,
                    ty: param_ty,
                });
                if !self.match_kind(&TokenKind::Comma) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen, "expected ')' after parameters")?;
        self.expect(&TokenKind::Colon, "expected ':' before return type")?;
        let return_type = self.parse_type()?;
        let effects = if self.match_kind(&TokenKind::Bang) {
            Some(self.parse_effect_row()?)
        } else {
            None
        };
        let span = span_join(start.span, self.previous_span());
        Ok(TopDecl::ExternDecl(ExternDecl {
            span,
            convention,
            safety,
            name,
            params,
            return_type,
            effects,
        }))
    }
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        if self.check(&TokenKind::Let) {
            return self.parse_let_expr();
        }
        if self.check(&TokenKind::If) {
            return self.parse_if_expr();
        }
        if self.check(&TokenKind::Lambda) {
            return self.parse_lambda_expr();
        }
        if self.check(&TokenKind::Handle) {
            return self.parse_handle_expr();
        }
        self.parse_bind_expr()
    }

    fn parse_let_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Let, "expected 'let'")?;
        let name = self.expect_ident("expected identifier after 'let'")?;
        let scheme = self.parse_optional_type_scheme()?;
        self.expect(&TokenKind::Equals, "expected '=' in let expression")?;
        let value = self.parse_expr()?;
        self.expect(&TokenKind::In, "expected 'in' after let binding")?;
        let body = self.parse_expr()?;
        let span = span_join(start.span, expr_span(&body));
        Ok(Expr::Let {
            span,
            name,
            scheme,
            value: Box::new(value),
            body: Box::new(body),
        })
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::If, "expected 'if'")?;
        let cond = self.parse_expr()?;
        self.expect(&TokenKind::Then, "expected 'then'")?;
        let then_branch = self.parse_expr()?;
        self.expect(&TokenKind::Else, "expected 'else'")?;
        let else_branch = self.parse_expr()?;
        let span = span_join(start.span, expr_span(&else_branch));
        Ok(Expr::If {
            span,
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        })
    }

    fn parse_lambda_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Lambda, "expected lambda")?;
        let param = self.expect_ident("expected parameter name")?;
        let param_type = if self.match_kind(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Arrow, "expected '->' in lambda")?;
        let body = self.parse_expr()?;
        let span = span_join(start.span, expr_span(&body));
        Ok(Expr::Lam {
            span,
            param,
            param_type,
            body: Box::new(body),
        })
    }

    fn parse_handle_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Handle, "expected 'handle'")?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::With, "expected 'with' after handle")?;
        let handler = self.parse_handler_ref()?;
        let span = span_join(start.span, handler_span(&handler));
        Ok(Expr::Handle {
            span,
            expr: Box::new(expr),
            handler,
        })
    }

    fn parse_handler_ref(&mut self) -> Result<HandlerRef, ParseError> {
        if self.check(&TokenKind::LBrace) {
            let def = self.parse_handler_def_block()?;
            return Ok(HandlerRef::Inline { span: def.span, def });
        }
        let name = self.expect_ident("expected handler name")?;
        let span = self.previous_span();
        Ok(HandlerRef::Named { span, name })
    }

    fn parse_bind_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_application()?;
        while self.match_kind(&TokenKind::Bind) {
            let right = self.parse_application()?;
            let span = span_join(expr_span(&expr), expr_span(&right));
            expr = Expr::RPMBind {
                span,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_application(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_postfix()?;
        while self.can_start_primary() {
            let arg = self.parse_postfix()?;
            let span = span_join(expr_span(&expr), expr_span(&arg));
            expr = Expr::App {
                span,
                func: Box::new(expr),
                arg: Box::new(arg),
            };
        }
        Ok(expr)
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        while self.match_kind(&TokenKind::Caret) {
            let index = self.parse_noetic_index()?;
            let span = span_join(expr_span(&expr), self.previous_span());
            expr = Expr::Noetic {
                span,
                index,
                expr: Box::new(expr),
            };
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        if self.check(&TokenKind::Return) {
            let start = self.advance().span;
            let expr = self.parse_postfix()?;
            let span = span_join(start, expr_span(&expr));
            return Ok(Expr::RPMReturn {
                span,
                expr: Box::new(expr),
            });
        }
        if self.check(&TokenKind::Check) {
            let start = self.advance().span;
            let expr = self.parse_postfix()?;
            let span = span_join(start, expr_span(&expr));
            return Ok(Expr::RPMCheck {
                span,
                expr: Box::new(expr),
            });
        }
        if self.check(&TokenKind::Acquire) {
            let start = self.advance().span;
            let expr = self.parse_postfix()?;
            let span = span_join(start, expr_span(&expr));
            return Ok(Expr::RPMAcquire {
                span,
                expr: Box::new(expr),
            });
        }
        if self.check(&TokenKind::Acbe) {
            return self.parse_acbe_expr();
        }
        if self.check(&TokenKind::Perform) {
            return self.parse_perform_expr();
        }
        if self.check(&TokenKind::Transfinite) {
            return self.parse_transfinite_loop();
        }
        if self.check(&TokenKind::Superpose) {
            return self.parse_superpose_expr();
        }
        if self.check(&TokenKind::Measure) {
            return self.parse_measure_expr();
        }
        if self.check(&TokenKind::Entangle) {
            return self.parse_entangle_expr();
        }
        if self.check(&TokenKind::FracOpen) {
            return self.parse_fractal_expr();
        }
        if !self.stop_at_pipe && self.check(&TokenKind::Pipe) {
            return self.parse_ket_expr();
        }
        if self.check(&TokenKind::LAngle) {
            if self.looks_like_fractal() {
                return self.parse_fractal_expr();
            }
            return self.parse_bra_or_braket();
        }
        if self.check(&TokenKind::LParen) {
            return self.parse_paren_or_unit_or_complex();
        }
        if matches!(self.peek_kind(), TokenKind::Noetic { .. } | TokenKind::Nu) {
            return self.parse_noetic_expr();
        }
        if matches!(self.peek_kind(), TokenKind::Element { .. }) {
            return self.parse_element_expr();
        }
        if matches!(self.peek_kind(), TokenKind::Foundation { .. }) {
            return self.parse_foundation_expr();
        }
        if matches!(
            self.peek_kind(),
            TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::Complex { .. }
                | TokenKind::Bool(_)
        ) {
            return self.parse_literal_expr();
        }
        if matches!(
            self.peek_kind(),
            TokenKind::Omega
                | TokenKind::Epsilon
                | TokenKind::Aleph
                | TokenKind::Succ
                | TokenKind::Limit
                | TokenKind::Ord
        ) {
            return self.parse_ordinal_expr();
        }
        if matches!(
            self.peek_kind(),
            TokenKind::Ident(_)
                | TokenKind::Amplitude
                | TokenKind::Basis
                | TokenKind::Resume
                | TokenKind::QState
        ) {
            return self.parse_var_expr();
        }
        Err(self.error_here("unexpected token in expression"))
    }
    fn parse_acbe_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Acbe, "expected 'acbe'")?;
        self.expect(&TokenKind::LParen, "expected '(' after 'acbe'")?;
        let goal = self.parse_expr()?;
        self.expect(&TokenKind::Comma, "expected ',' after acbe goal")?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after acbe expression")?;
        let span = span_join(start.span, expr_span(&expr));
        Ok(Expr::ACBE {
            span,
            goal: Box::new(goal),
            expr: Box::new(expr),
        })
    }

    fn parse_perform_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Perform, "expected 'perform'")?;
        let op = self.expect_ident("expected operation name")?;
        self.expect(&TokenKind::LParen, "expected '(' after operation name")?;
        let arg = if self.check(&TokenKind::RParen) {
            Expr::Lit {
                span: start.span,
                literal: Literal::Unit,
            }
        } else {
            self.parse_expr()?
        };
        self.expect(&TokenKind::RParen, "expected ')' after perform argument")?;
        let span = span_join(start.span, expr_span(&arg));
        Ok(Expr::Perform {
            span,
            op,
            arg: Box::new(arg),
        })
    }

    fn parse_transfinite_loop(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Transfinite, "expected 'transfinite'")?;
        self.expect(&TokenKind::Loop, "expected 'loop' after 'transfinite'")?;
        let index = self.expect_ident("expected loop index")?;
        self.expect(&TokenKind::LAngle, "expected '<' after loop index")?;
        let ordinal = self.parse_ordinal_expr()?;
        self.expect(&TokenKind::From, "expected 'from' in transfinite loop")?;
        let init = self.parse_expr()?;
        self.expect(&TokenKind::Step, "expected 'step' in transfinite loop")?;
        self.expect(&TokenKind::LParen, "expected '(' after step")?;
        let step_param = self.expect_ident("expected step parameter")?;
        self.expect(&TokenKind::Arrow, "expected '->' in step clause")?;
        let step_body = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after step clause")?;
        self.expect(&TokenKind::Limit, "expected 'limit' in transfinite loop")?;
        self.expect(&TokenKind::LParen, "expected '(' after limit")?;
        let limit_param = self.expect_ident("expected limit parameter")?;
        self.expect(&TokenKind::Arrow, "expected '->' in limit clause")?;
        let limit_body = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after limit clause")?;
        let span = span_join(start.span, expr_span(&limit_body));
        Ok(Expr::TransfiniteLoop {
            span,
            index,
            ordinal: Box::new(ordinal),
            init: Box::new(init),
            step_param,
            step_body: Box::new(step_body),
            limit_param,
            limit_body: Box::new(limit_body),
        })
    }

    fn parse_superpose_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Superpose, "expected 'superpose'")?;
        let mut states = Vec::new();
        let end_span;
        if self.match_kind(&TokenKind::LBrace) {
            if !self.check(&TokenKind::RBrace) {
                loop {
                    let amp = self.parse_expr()?;
                    self.expect(&TokenKind::Colon, "expected ':' after amplitude")?;
                    let ket = self.parse_ket_expr()?;
                    states.push((amp, ket));
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            let end = self.expect(&TokenKind::RBrace, "expected '}' after superpose list")?;
            end_span = end.span;
        } else if self.match_kind(&TokenKind::LBracket) {
            if !self.check(&TokenKind::RBracket) {
                loop {
                    self.expect(&TokenKind::LParen, "expected '(' in superpose pair")?;
                    let amp = self.parse_expr()?;
                    self.expect(&TokenKind::Comma, "expected ',' in superpose pair")?;
                    let val = self.parse_expr()?;
                    self.expect(&TokenKind::RParen, "expected ')' in superpose pair")?;
                    states.push((amp, val));
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            let end = self.expect(&TokenKind::RBracket, "expected ']' after superpose list")?;
            end_span = end.span;
        } else {
            return Err(self.error_here("expected '{' or '[' after 'superpose'"));
        }
        let span = span_join(start.span, end_span);
        Ok(Expr::Superpose { span, states })
    }

    fn parse_measure_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Measure, "expected 'measure'")?;
        self.expect(&TokenKind::LParen, "expected '(' after measure")?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after measure")?;
        let span = span_join(start.span, expr_span(&expr));
        Ok(Expr::Measure {
            span,
            expr: Box::new(expr),
        })
    }

    fn parse_entangle_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Entangle, "expected 'entangle'")?;
        self.expect(&TokenKind::LParen, "expected '(' after entangle")?;
        let left = self.parse_expr()?;
        self.expect(&TokenKind::Comma, "expected ',' in entangle")?;
        let right = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after entangle")?;
        let span = span_join(start.span, expr_span(&right));
        Ok(Expr::Entangle {
            span,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn parse_fractal_expr(&mut self) -> Result<Expr, ParseError> {
        let (start_span, close_kind) = if self.match_kind(&TokenKind::FracOpen) {
            (self.previous_span(), TokenKind::FracClose)
        } else {
            let start = self.expect(&TokenKind::LAngle, "expected '<'")?;
            (start.span, TokenKind::RAngle)
        };
        let mut seq = Vec::new();
        let first = self.parse_noetic_digit()?;
        seq.push(first);
        let mut ellipsis = None;
        loop {
            if self.match_kind(&TokenKind::Colon) {
                if self.check(&TokenKind::Dot) {
                    self.expect(&TokenKind::Dot, "expected '.' in ellipsis")?;
                    self.expect(&TokenKind::Dot, "expected '.' in ellipsis")?;
                    self.expect(&TokenKind::Dot, "expected '.' in ellipsis")?;
                    ellipsis = seq.last().copied();
                    break;
                }
                let next = self.parse_noetic_digit()?;
                seq.push(next);
                continue;
            }
            break;
        }
        self.expect(&close_kind, "expected fractal close")?;
        let subscript = if self.match_kind(&TokenKind::Underscore) {
            Some(Box::new(self.parse_ordinal_expr()?))
        } else {
            None
        };
        self.expect(&TokenKind::LParen, "expected '(' after fractal")?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after fractal argument")?;
        let span = span_join(start_span, expr_span(&expr));
        Ok(Expr::Fractal {
            span,
            seq,
            ellipsis,
            subscript,
            expr: Box::new(expr),
        })
    }

    fn parse_ket_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::Pipe, "expected '|' for ket")?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::RAngle, "expected '>' to close ket")?;
        let span = span_join(start.span, expr_span(&expr));
        Ok(Expr::Ket {
            span,
            expr: Box::new(expr),
        })
    }

    fn parse_bra_or_braket(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect(&TokenKind::LAngle, "expected '<' for bra")?;
        let prev = self.stop_at_pipe;
        self.stop_at_pipe = true;
        let left = self.parse_expr();
        self.stop_at_pipe = prev;
        let left = left?;
        self.expect(&TokenKind::Pipe, "expected '|' in bra")?;
        if self.check(&TokenKind::RAngle) {
            let end = self.expect(&TokenKind::RAngle, "expected '>' after bra")?;
            let span = span_join(start.span, end.span);
            return Ok(Expr::Bra {
                span,
                expr: Box::new(left),
            });
        }
        if !self.can_start_primary() {
            let span = span_join(start.span, expr_span(&left));
            return Ok(Expr::Bra {
                span,
                expr: Box::new(left),
            });
        }
        let right = self.parse_expr()?;
        self.expect(&TokenKind::RAngle, "expected '>' after bra-ket")?;
        let span = span_join(start.span, expr_span(&right));
        Ok(Expr::BraKet {
            span,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn parse_paren_or_unit_or_complex(&mut self) -> Result<Expr, ParseError> {
        if self.is_complex_pair() {
            let start = self.expect(&TokenKind::LParen, "expected '('")?;
            let re = self.parse_number_as_f64()?;
            self.expect(&TokenKind::Comma, "expected ',' in complex literal")?;
            let im = self.parse_number_as_f64()?;
            let end = self.expect(&TokenKind::RParen, "expected ')' after complex literal")?;
            let span = span_join(start.span, end.span);
            return Ok(Expr::Lit {
                span,
                literal: Literal::Complex { re, im },
            });
        }
        let start = self.expect(&TokenKind::LParen, "expected '('")?;
        if self.check(&TokenKind::RParen) {
            let end = self.expect(&TokenKind::RParen, "expected ')'")?;
            let span = span_join(start.span, end.span);
            return Ok(Expr::Lit {
                span,
                literal: Literal::Unit,
            });
        }
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "expected ')' after expression")?;
        Ok(expr)
    }

    fn parse_noetic_expr(&mut self) -> Result<Expr, ParseError> {
        let (start_span, index) = match self.peek_kind() {
            TokenKind::Noetic { index } => {
                let span = self.advance().span;
                (span, *index)
            }
            TokenKind::Nu => {
                let span = self.advance().span;
                let idx = self.parse_noetic_index()?;
                (span, idx)
            }
            _ => return Err(self.error_here("expected noetic operator")),
        };
        let expr = if self.match_kind(&TokenKind::LParen) {
            let inner = self.parse_expr()?;
            self.expect(&TokenKind::RParen, "expected ')' after noetic")?;
            inner
        } else {
            self.parse_postfix()?
        };
        let span = span_join(start_span, expr_span(&expr));
        Ok(Expr::Noetic {
            span,
            index,
            expr: Box::new(expr),
        })
    }

    fn parse_element_expr(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance().clone();
        if let TokenKind::Element { world, index } = token.kind {
            return Ok(Expr::Element {
                span: token.span,
                world,
                index,
            });
        }
        Err(self.error_here("expected element literal"))
    }

    fn parse_foundation_expr(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance().clone();
        if let TokenKind::Foundation { level, aspect } = token.kind {
            return Ok(Expr::Foundation {
                span: token.span,
                level,
                aspect,
            });
        }
        Err(self.error_here("expected foundation literal"))
    }

    fn parse_literal_expr(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance().clone();
        let span = token.span;
        let literal = match token.kind {
            TokenKind::Int(value) => Literal::Int(value),
            TokenKind::Bool(value) => Literal::Bool(value),
            TokenKind::Float(value) => Literal::Float(value),
            TokenKind::Complex { re, im } => Literal::Complex { re, im },
            _ => return Err(self.error_here("expected literal")),
        };
        Ok(Expr::Lit { span, literal })
    }

    fn parse_var_expr(&mut self) -> Result<Expr, ParseError> {
        let (name, span) = match self.peek_kind() {
            TokenKind::Ident(value) => (value.clone(), self.advance().span),
            TokenKind::Amplitude => ("amplitude".to_string(), self.advance().span),
            TokenKind::Basis => ("basis".to_string(), self.advance().span),
            TokenKind::Resume => ("resume".to_string(), self.advance().span),
            TokenKind::QState => ("qstate".to_string(), self.advance().span),
            _ => return Err(self.error_here("expected identifier")),
        };
        Ok(Expr::Var { span, name })
    }

    fn parse_number_as_f64(&mut self) -> Result<f64, ParseError> {
        match self.peek_kind() {
            TokenKind::Int(value) => {
                let value = *value as f64;
                self.advance();
                Ok(value)
            }
            TokenKind::Float(value) => {
                let value = *value;
                self.advance();
                Ok(value)
            }
            _ => Err(self.error_here("expected number")),
        }
    }

    fn parse_noetic_index(&mut self) -> Result<u8, ParseError> {
        let value = match self.peek_kind() {
            TokenKind::Int(value) => *value,
            _ => return Err(self.error_here("expected noetic index")),
        };
        if !(0..=9).contains(&value) {
            return Err(self.error_here("noetic index must be 0-9"));
        }
        self.advance();
        Ok(value as u8)
    }

    fn parse_noetic_digit(&mut self) -> Result<u8, ParseError> {
        let value = match self.peek_kind() {
            TokenKind::Int(value) => *value,
            _ => return Err(self.error_here("expected noetic digit")),
        };
        if !(0..=9).contains(&value) {
            return Err(self.error_here("fractal digit must be 0-9"));
        }
        self.advance();
        Ok(value as u8)
    }

    fn parse_ordinal_index(&mut self) -> Result<u64, ParseError> {
        let value = match self.peek_kind() {
            TokenKind::Int(value) => *value,
            _ => return Err(self.error_here("expected ordinal index")),
        };
        if value < 0 {
            return Err(self.error_here("ordinal index must be non-negative"));
        }
        self.advance();
        Ok(value as u64)
    }

    fn parse_ordinal_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_ordinal_add()
    }

    fn parse_ordinal_add(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_ordinal_mul()?;
        while self.match_kind(&TokenKind::Plus) {
            let right = self.parse_ordinal_mul()?;
            let span = span_join(expr_span(&expr), expr_span(&right));
            expr = Expr::OrdAdd {
                span,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_ordinal_mul(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_ordinal_exp()?;
        while self.match_kind(&TokenKind::Times) {
            let right = self.parse_ordinal_exp()?;
            let span = span_join(expr_span(&expr), expr_span(&right));
            expr = Expr::OrdMul {
                span,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_ordinal_exp(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_ordinal_primary()?;
        if self.match_kind(&TokenKind::Caret) {
            let right = self.parse_ordinal_exp()?;
            let span = span_join(expr_span(&left), expr_span(&right));
            return Ok(Expr::OrdExp {
                span,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_ordinal_primary(&mut self) -> Result<Expr, ParseError> {
        if self.match_kind(&TokenKind::Omega) {
            let span = self.previous_span();
            return Ok(Expr::OrdOmega { span });
        }
        if self.match_kind(&TokenKind::Epsilon) {
            let start = self.previous_span();
            let index = if self.match_kind(&TokenKind::Underscore) {
                self.parse_ordinal_index()?
            } else {
                0
            };
            let span = span_join(start, self.previous_span());
            return Ok(Expr::OrdEpsilon { span, index });
        }
        if self.match_kind(&TokenKind::Aleph) {
            let start = self.previous_span();
            let index = if self.match_kind(&TokenKind::Underscore) {
                self.parse_ordinal_index()?
            } else {
                0
            };
            let span = span_join(start, self.previous_span());
            return Ok(Expr::OrdAleph { span, index });
        }
        if self.match_kind(&TokenKind::Succ) {
            let start = self.previous_span();
            self.expect(&TokenKind::LParen, "expected '(' after succ")?;
            let expr = self.parse_ordinal_expr()?;
            self.expect(&TokenKind::RParen, "expected ')' after succ")?;
            let span = span_join(start, expr_span(&expr));
            return Ok(Expr::OrdSucc {
                span,
                expr: Box::new(expr),
            });
        }
        if self.match_kind(&TokenKind::Limit) {
            let start_span = self.previous_span();
            if self.match_kind(&TokenKind::LParen) {
                let binder = self.expect_ident("expected binder in limit")?;
                self.expect(&TokenKind::Arrow, "expected '->' in limit")?;
                let body = self.parse_ordinal_expr()?;
                self.expect(&TokenKind::RParen, "expected ')' after limit")?;
                let bound = Expr::OrdOmega { span: start_span };
                let span = span_join(start_span, expr_span(&body));
                return Ok(Expr::OrdLimit {
                    span,
                    binder,
                    bound: Box::new(bound),
                    body: Box::new(body),
                });
            }
            let binder = self.expect_ident("expected binder in limit")?;
            self.expect(&TokenKind::LAngle, "expected '<' in limit ordinal")?;
            let bound = self.parse_ordinal_expr()?;
            self.expect(&TokenKind::Dot, "expected '.' in limit ordinal")?;
            let body = self.parse_ordinal_expr()?;
            let span = span_join(start_span, expr_span(&body));
            return Ok(Expr::OrdLimit {
                span,
                binder,
                bound: Box::new(bound),
                body: Box::new(body),
            });
        }
        if self.match_kind(&TokenKind::Ord) {
            self.expect(&TokenKind::LParen, "expected '(' after ord")?;
            let expr = self.parse_ordinal_expr()?;
            self.expect(&TokenKind::RParen, "expected ')' after ord")?;
            return Ok(expr);
        }
        if self.match_kind(&TokenKind::LParen) {
            let expr = self.parse_ordinal_expr()?;
            self.expect(&TokenKind::RParen, "expected ')' after ordinal")?;
            return Ok(expr);
        }
        if let TokenKind::Int(value) = self.peek_kind() {
            let value = *value;
            let token = self.advance().clone();
            let ordinal = if value < 0 {
                return Err(self.error_here("ordinal literal must be non-negative"));
            } else {
                OrdinalLiteral::Finite(value as u64)
            };
            return Ok(Expr::OrdLit {
                span: token.span,
                value: ordinal,
            });
        }
        if let TokenKind::Ident(name) = self.peek_kind() {
            let name = name.clone();
            let span = self.advance().span;
            return Ok(Expr::Var { span, name });
        }
        Err(self.error_here("expected ordinal expression"))
    }
    fn parse_type(&mut self) -> Result<Type, ParseError> {
        self.parse_type_arrow()
    }

    fn parse_type_arrow(&mut self) -> Result<Type, ParseError> {
        let left = self.parse_type_effect()?;
        if self.match_kind(&TokenKind::Arrow) {
            let right = self.parse_type_arrow()?;
            return Ok(Type::Fun(Box::new(left), Box::new(right)));
        }
        Ok(left)
    }

    fn parse_type_effect(&mut self) -> Result<Type, ParseError> {
        let ty = self.parse_type_sum()?;
        if self.match_kind(&TokenKind::Bang) {
            let row = self.parse_effect_row()?;
            return Ok(Type::Effectful(Box::new(ty), row));
        }
        Ok(ty)
    }

    fn parse_type_sum(&mut self) -> Result<Type, ParseError> {
        let mut ty = self.parse_type_product()?;
        while self.match_kind(&TokenKind::Plus) {
            let right = self.parse_type_product()?;
            ty = Type::Sum(Box::new(ty), Box::new(right));
        }
        Ok(ty)
    }

    fn parse_type_product(&mut self) -> Result<Type, ParseError> {
        let mut ty = self.parse_type_atom()?;
        while self.match_kind(&TokenKind::Times) {
            let right = self.parse_type_atom()?;
            ty = Type::Product(Box::new(ty), Box::new(right));
        }
        Ok(ty)
    }

    fn parse_type_atom(&mut self) -> Result<Type, ParseError> {
        if self.match_kind(&TokenKind::LParen) {
            let ty = self.parse_type()?;
            self.expect(&TokenKind::RParen, "expected ')' after type")?;
            return Ok(ty);
        }

        if self.match_kind(&TokenKind::Handler) {
            return self.parse_handler_type_impl();
        }

        if self.match_kind(&TokenKind::QState) {
            let inner = self.parse_type_arg("qstate")?;
            return Ok(Type::QState(Box::new(inner)));
        }

        if let Some(name) = self.match_ident() {
            match name.as_str() {
                "Int" => return Ok(Type::Int),
                "Bool" => return Ok(Type::Bool),
                "Unit" => return Ok(Type::Unit),
                "Void" => return Ok(Type::Void),
                "Domain" => return Ok(Type::Domain),
                "Foundation" => return Ok(Type::Foundation),
                "Ord" | "Ordinal" => return Ok(Type::Ordinal),
                "Element" => {
                    let world = self.parse_world_param()?;
                    return Ok(Type::Element(world));
                }
                "Noetic" => {
                    let inner = self.parse_type_arg("Noetic")?;
                    return Ok(Type::Noetic(Box::new(inner)));
                }
                "Fractal" => {
                    let inner = self.parse_type_arg("Fractal")?;
                    return Ok(Type::Fractal(Box::new(inner)));
                }
                "RPM" => {
                    let inner = self.parse_type_arg("RPM")?;
                    return Ok(Type::RPM(Box::new(inner)));
                }
                "QState" => {
                    let inner = self.parse_type_arg("QState")?;
                    return Ok(Type::QState(Box::new(inner)));
                }
                "Handler" => {
                    return self.parse_handler_type_impl();
                }
                _ => return Ok(Type::Var(name)),
            }
        }

        Err(self.error_here("expected type"))
    }

    fn parse_handler_type_impl(&mut self) -> Result<Type, ParseError> {
        self.expect(&TokenKind::LBracket, "expected '[' after Handler")?;
        let effect = if self.check(&TokenKind::LBrace) {
            self.parse_effect_row()?
        } else {
            let name = self.expect_ident("expected effect name")?;
            EffectRow::Cons(name, Box::new(EffectRow::Empty))
        };
        self.expect(&TokenKind::Comma, "expected ',' after handler effect")?;
        let input = self.parse_type()?;
        self.expect(&TokenKind::Comma, "expected ',' after handler input")?;
        let output = self.parse_type()?;
        self.expect(&TokenKind::RBracket, "expected ']' after handler type")?;
        Ok(Type::Handler {
            effect,
            input: Box::new(input),
            output: Box::new(output),
        })
    }

    fn parse_type_arg(&mut self, name: &str) -> Result<Type, ParseError> {
        let message = format!("expected '[' after {}", name);
        self.expect(&TokenKind::LBracket, &message)?;
        let ty = self.parse_type()?;
        self.expect(&TokenKind::RBracket, "expected ']' after type argument")?;
        Ok(ty)
    }

    fn parse_world_param(&mut self) -> Result<Option<World>, ParseError> {
        if !self.match_kind(&TokenKind::LBracket) {
            return Ok(None);
        }
        let world_name = self.expect_ident("expected world name")?;
        self.expect(&TokenKind::RBracket, "expected ']' after world")?;
        let world = match world_name.as_str() {
            "A" => Some(World::A),
            "B" => Some(World::B),
            "C" => Some(World::C),
            "D" => Some(World::D),
            _ => None,
        };
        Ok(world)
    }

    fn parse_effect_row(&mut self) -> Result<EffectRow, ParseError> {
        self.expect(&TokenKind::LBrace, "expected '{' to start effect row")?;
        if self.match_kind(&TokenKind::RBrace) {
            return Ok(EffectRow::Empty);
        }
        let mut effects = Vec::new();
        loop {
            effects.push(self.expect_ident("expected effect name")?);
            if !self.match_kind(&TokenKind::Comma) {
                break;
            }
        }
        let tail = if self.match_kind(&TokenKind::Pipe) {
            let var = self.expect_ident("expected row variable")?;
            self.expect(&TokenKind::RBrace, "expected '}' after effect row")?;
            EffectRow::Var(var)
        } else {
            self.expect(&TokenKind::RBrace, "expected '}' after effect row")?;
            EffectRow::Empty
        };
        let mut row = tail;
        for name in effects.into_iter().rev() {
            row = EffectRow::Cons(name, Box::new(row));
        }
        Ok(row)
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .expect("parser out of bounds")
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn peek_kind_n(&self, n: usize) -> Option<&TokenKind> {
        self.tokens.get(self.pos + n).map(|token| &token.kind)
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.pos += 1;
        }
        self.previous()
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.pos - 1]
    }

    fn previous_span(&self) -> Span {
        self.previous().span
    }

    fn check(&self, kind: &TokenKind) -> bool {
        same_kind(self.peek_kind(), kind)
    }

    fn match_kind(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind, message: &str) -> Result<Token, ParseError> {
        if self.check(kind) {
            Ok(self.advance().clone())
        } else {
            Err(self.error_here(message))
        }
    }

    fn expect_ident(&mut self, message: &str) -> Result<Ident, ParseError> {
        if let TokenKind::Ident(name) = self.peek_kind() {
            let name = name.clone();
            self.advance();
            Ok(name)
        } else {
            Err(self.error_here(message))
        }
    }

    fn match_ident(&mut self) -> Option<Ident> {
        if let TokenKind::Ident(name) = self.peek_kind() {
            let name = name.clone();
            self.advance();
            Some(name)
        } else {
            None
        }
    }

    fn error_here(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(message, Some(self.peek().span))
    }

    fn can_start_primary(&self) -> bool {
        let can = matches!(
            self.peek_kind(),
            TokenKind::Ident(_)
                | TokenKind::Int(_)
                | TokenKind::Bool(_)
                | TokenKind::Float(_)
                | TokenKind::Complex { .. }
                | TokenKind::Element { .. }
                | TokenKind::Foundation { .. }
                | TokenKind::Noetic { .. }
                | TokenKind::Nu
                | TokenKind::FracOpen
                | TokenKind::LParen
                | TokenKind::LAngle
                | TokenKind::Pipe
                | TokenKind::Return
                | TokenKind::Check
                | TokenKind::Acquire
                | TokenKind::Acbe
                | TokenKind::Perform
                | TokenKind::Transfinite
                | TokenKind::Superpose
                | TokenKind::Measure
                | TokenKind::Entangle
                | TokenKind::Omega
                | TokenKind::Epsilon
                | TokenKind::Aleph
                | TokenKind::Succ
                | TokenKind::Limit
                | TokenKind::Ord
                | TokenKind::Amplitude
                | TokenKind::Basis
                | TokenKind::Resume
                | TokenKind::QState
        );
        if self.stop_at_pipe && matches!(self.peek_kind(), TokenKind::Pipe) {
            return false;
        }
        can
    }

    fn looks_like_fractal(&self) -> bool {
        match self.peek_kind_n(1) {
            Some(TokenKind::Int(_)) => {}
            _ => return false,
        }
        let mut idx = self.pos + 1;
        while let Some(token) = self.tokens.get(idx) {
            match token.kind {
                TokenKind::Pipe => return false,
                TokenKind::RAngle => return true,
                TokenKind::Eof => return false,
                _ => idx += 1,
            }
        }
        false
    }

    fn is_complex_pair(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::LParen)
            && matches!(self.peek_kind_n(1), Some(TokenKind::Int(_) | TokenKind::Float(_)))
            && matches!(self.peek_kind_n(2), Some(TokenKind::Comma))
            && matches!(self.peek_kind_n(3), Some(TokenKind::Int(_) | TokenKind::Float(_)))
            && matches!(self.peek_kind_n(4), Some(TokenKind::RParen))
    }
}

fn same_kind(a: &TokenKind, b: &TokenKind) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn span_join(start: Span, end: Span) -> Span {
    Span::new(start.start, end.end, start.line, start.column)
}

fn expr_span(expr: &Expr) -> Span {
    match expr {
        Expr::Var { span, .. }
        | Expr::Lit { span, .. }
        | Expr::Lam { span, .. }
        | Expr::App { span, .. }
        | Expr::Let { span, .. }
        | Expr::If { span, .. }
        | Expr::Element { span, .. }
        | Expr::Foundation { span, .. }
        | Expr::Noetic { span, .. }
        | Expr::Fractal { span, .. }
        | Expr::RPMReturn { span, .. }
        | Expr::RPMBind { span, .. }
        | Expr::RPMCheck { span, .. }
        | Expr::RPMAcquire { span, .. }
        | Expr::ACBE { span, .. }
        | Expr::Handle { span, .. }
        | Expr::Perform { span, .. }
        | Expr::OrdLit { span, .. }
        | Expr::OrdOmega { span, .. }
        | Expr::OrdEpsilon { span, .. }
        | Expr::OrdAleph { span, .. }
        | Expr::OrdSucc { span, .. }
        | Expr::OrdAdd { span, .. }
        | Expr::OrdMul { span, .. }
        | Expr::OrdExp { span, .. }
        | Expr::OrdLimit { span, .. }
        | Expr::TransfiniteLoop { span, .. }
        | Expr::Superpose { span, .. }
        | Expr::Measure { span, .. }
        | Expr::Entangle { span, .. }
        | Expr::Ket { span, .. }
        | Expr::Bra { span, .. }
        | Expr::BraKet { span, .. } => *span,
    }
}

fn handler_span(handler: &HandlerRef) -> Span {
    match handler {
        HandlerRef::Named { span, .. } | HandlerRef::Inline { span, .. } => *span,
    }
}

fn build_param_type(params: Vec<Type>) -> Type {
    let mut iter = params.into_iter();
    let Some(first) = iter.next() else {
        return Type::Unit;
    };
    iter.fold(first, |acc, ty| Type::Product(Box::new(acc), Box::new(ty)))
}
