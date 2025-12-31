use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{
    Convention, EffectRow, ExportDecl, ExportItem, ExternDecl, ExternParam, ImportDecl, ImportItem,
    ModuleBody, ModuleDecl, OpSig, Program, Safety, TopDecl, Type,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedProgram {
    pub modules: Vec<ResolvedModule>,
}

impl ResolvedProgram {
    pub fn module(&self, path: &[&str]) -> Option<&ResolvedModule> {
        self.modules
            .iter()
            .find(|module| module.path.iter().map(String::as_str).eq(path.iter().copied()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModule {
    pub path: Vec<String>,
    pub exports: ModuleExports,
    pub scope: ModuleScope,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleExports {
    pub values: Vec<String>,
    pub types: Vec<ExportedType>,
    pub effects: Vec<EffectSignature>,
    pub externs: Vec<ExternSignature>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedType {
    pub name: String,
    pub transparent: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectSignature {
    pub name: String,
    pub ops: Vec<EffectOpSignature>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectOpSignature {
    pub name: String,
    pub input: Type,
    pub output: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternSignature {
    pub name: String,
    pub convention: Convention,
    pub safety: Safety,
    pub params: Vec<ExternParamSignature>,
    pub return_type: Type,
    pub effects: Option<EffectRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternParamSignature {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleScope {
    pub values: BTreeSet<String>,
    pub types: BTreeSet<String>,
    pub modules: BTreeMap<String, Vec<String>>,
    pub effects: BTreeMap<String, EffectSignature>,
    pub externs: BTreeMap<String, ExternSignature>,
}

impl ModuleScope {
    fn new() -> Self {
        Self {
            values: BTreeSet::new(),
            types: BTreeSet::new(),
            modules: BTreeMap::new(),
            effects: BTreeMap::new(),
            externs: BTreeMap::new(),
        }
    }

    fn insert_value(&mut self, name: String) -> Result<(), ResolveError> {
        if !self.values.insert(name.clone()) {
            return Err(ResolveError::DuplicateImport(name));
        }
        Ok(())
    }

    fn insert_type(&mut self, name: String) -> Result<(), ResolveError> {
        if !self.types.insert(name.clone()) {
            return Err(ResolveError::DuplicateImport(name));
        }
        Ok(())
    }

    fn insert_module(&mut self, alias: String, path: Vec<String>) -> Result<(), ResolveError> {
        if self.modules.contains_key(&alias) {
            return Err(ResolveError::DuplicateImport(alias));
        }
        self.modules.insert(alias, path);
        Ok(())
    }

    fn insert_effect(&mut self, effect: EffectSignature) -> Result<(), ResolveError> {
        if self.effects.contains_key(&effect.name) {
            return Err(ResolveError::DuplicateImport(effect.name));
        }
        self.effects.insert(effect.name.clone(), effect);
        Ok(())
    }

    fn insert_extern(&mut self, extern_sig: ExternSignature) -> Result<(), ResolveError> {
        if self.externs.contains_key(&extern_sig.name) {
            return Err(ResolveError::DuplicateImport(extern_sig.name));
        }
        self.externs
            .insert(extern_sig.name.clone(), extern_sig);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    DuplicateModule(String),
    DuplicateImport(String),
    UnknownModule(String),
    UnknownExport(String),
    UnknownImportItem { module: String, name: String },
    RecursiveModule(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::DuplicateModule(name) => write!(f, "duplicate module: {name}"),
            ResolveError::DuplicateImport(name) => write!(f, "duplicate import: {name}"),
            ResolveError::UnknownModule(name) => write!(f, "unknown module: {name}"),
            ResolveError::UnknownExport(name) => write!(f, "unknown export: {name}"),
            ResolveError::UnknownImportItem { module, name } => {
                write!(f, "unknown import '{name}' from {module}")
            }
            ResolveError::RecursiveModule(name) => write!(f, "recursive module: {name}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModuleStatus {
    Pending,
    InProgress,
    Done,
}

#[derive(Debug, Clone)]
struct ModuleEntry {
    body: ModuleBody,
    status: ModuleStatus,
    resolved: Option<ResolvedModule>,
    exports: Option<ModuleExports>,
}

#[derive(Debug, Default)]
struct ModuleTable {
    modules: BTreeMap<Vec<String>, ModuleEntry>,
}

#[derive(Debug, Clone)]
struct ModuleSymbols {
    values: BTreeSet<String>,
    types: BTreeSet<String>,
    modules: BTreeMap<String, Vec<String>>,
    effects: BTreeMap<String, EffectSignature>,
    externs: BTreeMap<String, ExternSignature>,
}

pub fn resolve_program(program: &Program) -> Result<ResolvedProgram, ResolveError> {
    let mut table = ModuleTable::default();
    collect_modules(&program.decls, &[], &mut table)?;
    let mut modules = Vec::new();
    let paths: Vec<Vec<String>> = table.modules.keys().cloned().collect();
    for path in paths {
        modules.push(resolve_module(&mut table, &path)?);
    }
    modules.sort_by(|left, right| module_path_string(&left.path).cmp(&module_path_string(&right.path)));
    Ok(ResolvedProgram { modules })
}

fn collect_modules(
    decls: &[TopDecl],
    prefix: &[String],
    table: &mut ModuleTable,
) -> Result<(), ResolveError> {
    for decl in decls {
        if let TopDecl::ModuleDecl(module) = decl {
            collect_module_decl(module, prefix, table)?;
        }
    }
    Ok(())
}

fn collect_module_decl(
    module: &ModuleDecl,
    prefix: &[String],
    table: &mut ModuleTable,
) -> Result<(), ResolveError> {
    let mut full_path = prefix.to_vec();
    full_path.extend(module.path.iter().cloned());
    let name = module_path_string(&full_path);
    if table.modules.contains_key(&full_path) {
        return Err(ResolveError::DuplicateModule(name));
    }
    table.modules.insert(
        full_path.clone(),
        ModuleEntry {
            body: module.body.clone(),
            status: ModuleStatus::Pending,
            resolved: None,
            exports: None,
        },
    );
    collect_modules(&module.body.decls, &full_path, table)?;
    Ok(())
}

fn resolve_module(table: &mut ModuleTable, path: &[String]) -> Result<ResolvedModule, ResolveError> {
    let (status, cached) = {
        let entry = table
            .modules
            .get(path)
            .ok_or_else(|| ResolveError::UnknownModule(module_path_string(path)))?;
        (entry.status, entry.resolved.clone())
    };

    match status {
        ModuleStatus::Done => return Ok(cached.expect("resolved module cached")),
        ModuleStatus::InProgress => {
            return Err(ResolveError::RecursiveModule(module_path_string(path)))
        }
        ModuleStatus::Pending => {}
    }

    if let Some(entry) = table.modules.get_mut(path) {
        entry.status = ModuleStatus::InProgress;
    }

    let body = table
        .modules
        .get(path)
        .ok_or_else(|| ResolveError::UnknownModule(module_path_string(path)))?
        .body
        .clone();

    let symbols = collect_symbols(path, &body);
    let mut scope = ModuleScope::new();
    for value in &symbols.values {
        scope.insert_value(value.clone())?;
    }
    for ty in &symbols.types {
        scope.insert_type(ty.clone())?;
    }
    for (alias, target) in &symbols.modules {
        scope.insert_module(alias.clone(), target.clone())?;
    }
    for effect in symbols.effects.values() {
        scope.insert_effect(effect.clone())?;
    }
    for extern_sig in symbols.externs.values() {
        scope.insert_extern(extern_sig.clone())?;
    }

    for import in &body.imports {
        apply_import(table, path, import, &mut scope)?;
    }

    let exports = resolve_exports(&body, &symbols, &scope)?;
    let resolved = ResolvedModule {
        path: path.to_vec(),
        exports: exports.clone(),
        scope,
    };

    if let Some(entry) = table.modules.get_mut(path) {
        entry.status = ModuleStatus::Done;
        entry.resolved = Some(resolved.clone());
        entry.exports = Some(exports);
    }

    Ok(resolved)
}

fn resolve_module_exports(
    table: &mut ModuleTable,
    path: &[String],
) -> Result<ModuleExports, ResolveError> {
    if let Some(entry) = table.modules.get(path) {
        if let Some(exports) = &entry.exports {
            return Ok(exports.clone());
        }
    }
    let resolved = resolve_module(table, path)?;
    Ok(resolved.exports)
}

fn collect_symbols(path: &[String], body: &ModuleBody) -> ModuleSymbols {
    let mut values = BTreeSet::new();
    let mut types = BTreeSet::new();
    let mut modules = BTreeMap::new();
    let mut effects = BTreeMap::new();
    let mut externs = BTreeMap::new();

    for decl in &body.decls {
        match decl {
            TopDecl::LetDecl { name, .. } => {
                values.insert(name.clone());
            }
            TopDecl::ClassDecl(class_decl) => {
                values.insert(class_decl.name.clone());
                types.insert(class_decl.name.clone());
            }
            TopDecl::ExternDecl(decl) => {
                externs.insert(decl.name.clone(), extern_signature(decl));
            }
            TopDecl::EffectDecl { name, ops, .. } => {
                effects.insert(name.clone(), effect_signature(name, ops));
            }
            TopDecl::HandlerDecl { name, .. } => {
                values.insert(name.clone());
            }
            TopDecl::TypeDecl { name, .. } => {
                types.insert(name.clone());
            }
            TopDecl::ModuleDecl(module) => {
                if let Some(name) = module.path.last() {
                    values.insert(name.clone());
                    let mut full_path = path.to_vec();
                    full_path.extend(module.path.iter().cloned());
                    modules.insert(name.clone(), full_path);
                }
            }
        }
    }

    ModuleSymbols {
        values,
        types,
        modules,
        effects,
        externs,
    }
}

fn resolve_exports(
    body: &ModuleBody,
    symbols: &ModuleSymbols,
    scope: &ModuleScope,
) -> Result<ModuleExports, ResolveError> {
    match &body.exports {
        Some(export_decl) => resolve_export_list(export_decl, scope),
        None => {
            let mut values: Vec<String> = symbols.values.iter().cloned().collect();
            values.sort();
            let mut types: Vec<ExportedType> = symbols
                .types
                .iter()
                .map(|name| ExportedType {
                    name: name.clone(),
                    transparent: true,
                })
                .collect();
            types.sort_by(|left, right| left.name.cmp(&right.name));
            let mut effects: Vec<EffectSignature> = symbols.effects.values().cloned().collect();
            effects.sort_by(|left, right| left.name.cmp(&right.name));
            let mut externs: Vec<ExternSignature> = symbols.externs.values().cloned().collect();
            externs.sort_by(|left, right| left.name.cmp(&right.name));
            Ok(ModuleExports {
                values,
                types,
                effects,
                externs,
            })
        }
    }
}

fn resolve_export_list(
    export_decl: &ExportDecl,
    scope: &ModuleScope,
) -> Result<ModuleExports, ResolveError> {
    let mut values = Vec::new();
    let mut types = Vec::new();
    let mut effects = Vec::new();
    let mut externs = Vec::new();

    for item in &export_decl.items {
        match item {
            ExportItem::Value(name) => {
                if let Some(effect) = scope.effects.get(name) {
                    effects.push(effect.clone());
                } else if let Some(extern_sig) = scope.externs.get(name) {
                    externs.push(extern_sig.clone());
                } else if scope.values.contains(name) {
                    values.push(name.clone());
                } else {
                    return Err(ResolveError::UnknownExport(name.clone()));
                }
            }
            ExportItem::Type { name, transparent } => {
                if !scope.types.contains(name) {
                    return Err(ResolveError::UnknownExport(name.clone()));
                }
                types.push(ExportedType {
                    name: name.clone(),
                    transparent: *transparent,
                });
            }
        }
    }

    values.sort();
    types.sort_by(|left, right| left.name.cmp(&right.name));
    effects.sort_by(|left, right| left.name.cmp(&right.name));
    externs.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(ModuleExports {
        values,
        types,
        effects,
        externs,
    })
}

fn apply_import(
    table: &mut ModuleTable,
    current_path: &[String],
    import: &ImportDecl,
    scope: &mut ModuleScope,
) -> Result<(), ResolveError> {
    match import {
        ImportDecl::Qualified { path, .. } => {
            let target = resolve_import_path(table, current_path, path)?;
            let alias = path
                .last()
                .cloned()
                .unwrap_or_else(|| module_path_string(&target));
            scope.insert_module(alias, target)?;
        }
        ImportDecl::Aliased { path, alias, .. } => {
            let target = resolve_import_path(table, current_path, path)?;
            scope.insert_module(alias.clone(), target)?;
        }
        ImportDecl::Wildcard { path, .. } => {
            let target = resolve_import_path(table, current_path, path)?;
            let exports = resolve_module_exports(table, &target)?;
            for name in exports.values {
                scope.insert_value(name)?;
            }
            for ty in exports.types {
                scope.insert_type(ty.name)?;
            }
            for effect in exports.effects {
                scope.insert_effect(effect)?;
            }
            for extern_sig in exports.externs {
                scope.insert_extern(extern_sig)?;
            }
        }
        ImportDecl::Selective { path, items, .. } => {
            let target = resolve_import_path(table, current_path, path)?;
            let exports = resolve_module_exports(table, &target)?;
            apply_selective_import(&exports, path, items, scope)?;
        }
    }
    Ok(())
}

fn apply_selective_import(
    exports: &ModuleExports,
    path: &[String],
    items: &[ImportItem],
    scope: &mut ModuleScope,
) -> Result<(), ResolveError> {
    let module_name = module_path_string(path);
    for item in items {
        let import_name = &item.name;
        let target_name = item.alias.as_ref().unwrap_or(import_name);
        if let Some(effect) = exports.effects.iter().find(|effect| effect.name == *import_name) {
            let mut effect_sig = effect.clone();
            effect_sig.name = target_name.clone();
            scope.insert_effect(effect_sig)?;
            continue;
        }
        if let Some(extern_sig) = exports
            .externs
            .iter()
            .find(|extern_sig| extern_sig.name == *import_name)
        {
            let mut extern_sig = extern_sig.clone();
            extern_sig.name = target_name.clone();
            scope.insert_extern(extern_sig)?;
            continue;
        }
        let is_value = exports.values.iter().any(|name| name == import_name);
        let ty = exports
            .types
            .iter()
            .find(|ty| ty.name == *import_name)
            .cloned();
        match (is_value, ty) {
            (true, None) => scope.insert_value(target_name.clone())?,
            (false, Some(_)) => scope.insert_type(target_name.clone())?,
            _ => {
                return Err(ResolveError::UnknownImportItem {
                    module: module_name,
                    name: import_name.clone(),
                })
            }
        }
    }
    Ok(())
}

fn resolve_import_path(
    table: &ModuleTable,
    current_path: &[String],
    path: &[String],
) -> Result<Vec<String>, ResolveError> {
    if table.modules.contains_key(path) {
        return Ok(path.to_vec());
    }
    let mut relative = current_path.to_vec();
    relative.extend(path.iter().cloned());
    if table.modules.contains_key(&relative) {
        return Ok(relative);
    }
    Err(ResolveError::UnknownModule(module_path_string(path)))
}

fn effect_signature(name: &str, ops: &[OpSig]) -> EffectSignature {
    let mut entries = Vec::new();
    for op in ops {
        entries.push(EffectOpSignature {
            name: op.name.clone(),
            input: op.input.clone(),
            output: op.output.clone(),
        });
    }
    EffectSignature {
        name: name.to_string(),
        ops: entries,
    }
}

fn extern_signature(decl: &ExternDecl) -> ExternSignature {
    ExternSignature {
        name: decl.name.clone(),
        convention: decl.convention.clone(),
        safety: decl.safety.clone(),
        params: extern_params(&decl.params),
        return_type: decl.return_type.clone(),
        effects: decl.effects.clone(),
    }
}

fn extern_params(params: &[ExternParam]) -> Vec<ExternParamSignature> {
    let mut out = Vec::new();
    for param in params {
        out.push(ExternParamSignature {
            name: param.name.clone(),
            ty: param.ty.clone(),
        });
    }
    out
}

fn module_path_string(path: &[String]) -> String {
    path.join(".")
}
