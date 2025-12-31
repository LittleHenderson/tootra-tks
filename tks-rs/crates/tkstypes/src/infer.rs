use std::collections::{HashMap, HashSet};

use tkscore::ast::{
    ClassDecl, EffectRow, ExportDecl, ExportItem, Expr, HandlerDef, HandlerRef, ImportDecl,
    ImportItem, Literal, ModuleBody, ModuleDecl, Program, TopDecl, Type, TypeScheme,
};

#[derive(Debug, Clone)]
pub enum TypeError {
    Mismatch(Type, Type),
    Occurs(String, Type),
    RowOccurs(String, EffectRow),
    RowMismatch(EffectRow, EffectRow),
    UnknownVar(String),
    UnknownOp(String),
    UnknownHandler(String),
    UnknownModule(String),
    UnknownClass(String),
    UnknownMember { class: String, member: String },
    MissingMember { class: String, member: String },
    DuplicateOp(String),
    DuplicateType(String),
    DuplicateClass(String),
    DuplicateMember { class: String, member: String },
    DuplicateModule(String),
    RecursiveTypeAlias(String),
    HandlerEffectMismatch { handler: String, expected: String, found: String },
    InvalidAnnotation(String),
    InvalidMemberAccess { member: String, ty: Type },
    InvalidMethodCall { class: String, method: String },
    Unimplemented(&'static str),
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::Mismatch(left, right) => {
                write!(f, "type mismatch: {left:?} vs {right:?}")
            }
            TypeError::Occurs(var, ty) => write!(f, "occurs check failed: {var} in {ty:?}"),
            TypeError::RowOccurs(var, row) => write!(f, "row occurs check failed: {var} in {row:?}"),
            TypeError::RowMismatch(left, right) => {
                write!(f, "effect row mismatch: {left:?} vs {right:?}")
            }
            TypeError::UnknownVar(name) => write!(f, "unknown variable: {name}"),
            TypeError::UnknownOp(name) => write!(f, "unknown effect operation: {name}"),
            TypeError::UnknownHandler(name) => write!(f, "unknown handler: {name}"),
            TypeError::UnknownModule(name) => write!(f, "unknown module: {name}"),
            TypeError::UnknownClass(name) => write!(f, "unknown class: {name}"),
            TypeError::UnknownMember { class, member } => {
                write!(f, "unknown member {member} on class {class}")
            }
            TypeError::MissingMember { class, member } => {
                write!(f, "missing member {member} on class {class}")
            }
            TypeError::DuplicateOp(name) => write!(f, "duplicate effect operation: {name}"),
            TypeError::DuplicateType(name) => write!(f, "duplicate type alias: {name}"),
            TypeError::DuplicateClass(name) => write!(f, "duplicate class: {name}"),
            TypeError::DuplicateMember { class, member } => {
                write!(f, "duplicate member {member} on class {class}")
            }
            TypeError::DuplicateModule(name) => write!(f, "duplicate module: {name}"),
            TypeError::RecursiveTypeAlias(name) => {
                write!(f, "recursive type alias: {name}")
            }
            TypeError::HandlerEffectMismatch {
                handler,
                expected,
                found,
            } => write!(
                f,
                "handler {handler} expected effect {expected}, found {found}"
            ),
            TypeError::InvalidAnnotation(message) => write!(f, "invalid annotation: {message}"),
            TypeError::InvalidMemberAccess { member, ty } => {
                write!(f, "invalid member access {member} on {ty:?}")
            }
            TypeError::InvalidMethodCall { class, method } => {
                write!(f, "invalid call to method {method} on class {class}")
            }
            TypeError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferOutput {
    pub ty: Type,
    pub effect: EffectRow,
}

#[derive(Debug, Clone)]
pub struct InferProgramOutput {
    pub env: TypeEnv,
    pub entry: Option<InferOutput>,
}

#[derive(Debug, Clone)]
pub struct OpSigEntry {
    pub effect: String,
    pub type_vars: Vec<String>,
    pub input: Type,
    pub output: Type,
}

#[derive(Debug, Clone)]
pub struct HandlerInfo {
    pub effect: String,
    pub type_vars: Vec<String>,
    pub row_vars: Vec<String>,
    pub input: Type,
    pub output: Type,
    pub effect_row: EffectRow,
}

impl HandlerInfo {
    fn instantiate(&self, state: &mut InferState) -> (Type, Type, EffectRow) {
        let mut subst = Subst::default();
        for var in &self.type_vars {
            subst.type_subst.insert(var.clone(), state.fresh_type_var());
        }
        for var in &self.row_vars {
            subst.row_subst.insert(var.clone(), state.fresh_row_var());
        }
        (
            subst.apply_type(&self.input),
            subst.apply_type(&self.output),
            subst.apply_row(&self.effect_row),
        )
    }
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub params: Vec<String>,
    pub body: Type,
}

#[derive(Debug, Clone)]
pub struct PropertySig {
    pub ty: Type,
    pub effect: EffectRow,
}

impl PropertySig {
    fn apply_subst(&self, subst: &Subst) -> Self {
        Self {
            ty: subst.apply_type(&self.ty),
            effect: subst.apply_row(&self.effect),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MethodSig {
    pub params: Vec<Type>,
    pub output: Type,
    pub effect: EffectRow,
}

impl MethodSig {
    fn apply_subst(&self, subst: &Subst) -> Self {
        Self {
            params: self
                .params
                .iter()
                .map(|param| subst.apply_type(param))
                .collect(),
            output: subst.apply_type(&self.output),
            effect: subst.apply_row(&self.effect),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ClassInfo {
    pub fields: HashMap<String, Type>,
    pub properties: HashMap<String, PropertySig>,
    pub methods: HashMap<String, MethodSig>,
}

impl ClassInfo {
    fn apply_subst(&self, subst: &Subst) -> Self {
        let fields = self
            .fields
            .iter()
            .map(|(name, ty)| (name.clone(), subst.apply_type(ty)))
            .collect();
        let properties = self
            .properties
            .iter()
            .map(|(name, sig)| (name.clone(), sig.apply_subst(subst)))
            .collect();
        let methods = self
            .methods
            .iter()
            .map(|(name, sig)| (name.clone(), sig.apply_subst(subst)))
            .collect();
        Self {
            fields,
            properties,
            methods,
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
    exports: Option<TypeEnv>,
}

#[derive(Debug, Default)]
struct ModuleTable {
    modules: HashMap<Vec<String>, ModuleEntry>,
}

#[derive(Debug, Clone, Default)]
pub struct Subst {
    type_subst: HashMap<String, Type>,
    row_subst: HashMap<String, EffectRow>,
}

impl Subst {
    pub fn apply_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(name) => self
                .type_subst
                .get(name)
                .map(|t| self.apply_type(t))
                .unwrap_or_else(|| Type::Var(name.clone())),
            Type::Class(name) => Type::Class(name.clone()),
            Type::Int => Type::Int,
            Type::Bool => Type::Bool,
            Type::Unit => Type::Unit,
            Type::Void => Type::Void,
            Type::Element(world) => Type::Element(*world),
            Type::Foundation => Type::Foundation,
            Type::Domain => Type::Domain,
            Type::Noetic(inner) => Type::Noetic(Box::new(self.apply_type(inner))),
            Type::Fractal(inner) => Type::Fractal(Box::new(self.apply_type(inner))),
            Type::RPM(inner) => Type::RPM(Box::new(self.apply_type(inner))),
            Type::Fun(left, right) => Type::Fun(
                Box::new(self.apply_type(left)),
                Box::new(self.apply_type(right)),
            ),
            Type::Product(left, right) => Type::Product(
                Box::new(self.apply_type(left)),
                Box::new(self.apply_type(right)),
            ),
            Type::Sum(left, right) => Type::Sum(
                Box::new(self.apply_type(left)),
                Box::new(self.apply_type(right)),
            ),
            Type::Effectful(inner, row) => {
                Type::Effectful(Box::new(self.apply_type(inner)), self.apply_row(row))
            }
            Type::Handler { effect, input, output } => Type::Handler {
                effect: self.apply_row(effect),
                input: Box::new(self.apply_type(input)),
                output: Box::new(self.apply_type(output)),
            },
            Type::Ordinal => Type::Ordinal,
            Type::QState(inner) => Type::QState(Box::new(self.apply_type(inner))),
        }
    }

    pub fn apply_row(&self, row: &EffectRow) -> EffectRow {
        match row {
            EffectRow::Empty => EffectRow::Empty,
            EffectRow::Cons(name, tail) => {
                EffectRow::Cons(name.clone(), Box::new(self.apply_row(tail)))
            }
            EffectRow::Var(name) => self
                .row_subst
                .get(name)
                .map(|r| self.apply_row(r))
                .unwrap_or_else(|| EffectRow::Var(name.clone())),
        }
    }

    pub fn row_binding(&self, name: &str) -> Option<EffectRow> {
        self.row_subst.get(name).cloned()
    }

    fn bind_type_var(&mut self, name: &str, ty: &Type) -> Result<(), TypeError> {
        if let Type::Var(existing) = ty {
            if existing == name {
                return Ok(());
            }
        }
        if type_contains_var(ty, name, self) {
            return Err(TypeError::Occurs(name.to_string(), ty.clone()));
        }
        self.type_subst.insert(name.to_string(), ty.clone());
        Ok(())
    }

    fn bind_row_var(&mut self, name: &str, row: &EffectRow) -> Result<(), TypeError> {
        if let EffectRow::Var(existing) = row {
            if existing == name {
                return Ok(());
            }
        }
        if row_contains_var(row, name, self) {
            return Err(TypeError::RowOccurs(name.to_string(), row.clone()));
        }
        self.row_subst.insert(name.to_string(), row.clone());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Scheme {
    type_vars: Vec<String>,
    row_vars: Vec<String>,
    ty: Type,
}

impl Scheme {
    pub fn mono(ty: Type) -> Self {
        Self {
            type_vars: Vec::new(),
            row_vars: Vec::new(),
            ty,
        }
    }

    fn instantiate(&self, state: &mut InferState) -> Type {
        let mut subst = Subst::default();
        for var in &self.type_vars {
            subst
                .type_subst
                .insert(var.clone(), state.fresh_type_var());
        }
        for var in &self.row_vars {
            subst
                .row_subst
                .insert(var.clone(), state.fresh_row_var());
        }
        subst.apply_type(&self.ty)
    }

    fn apply_subst(&self, subst: &Subst) -> Self {
        let mut filtered = subst.clone();
        for var in &self.type_vars {
            filtered.type_subst.remove(var);
        }
        for var in &self.row_vars {
            filtered.row_subst.remove(var);
        }
        Self {
            type_vars: self.type_vars.clone(),
            row_vars: self.row_vars.clone(),
            ty: filtered.apply_type(&self.ty),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    values: HashMap<String, Scheme>,
    ops: HashMap<String, OpSigEntry>,
    handlers: HashMap<String, HandlerInfo>,
    type_aliases: HashMap<String, TypeAlias>,
    classes: HashMap<String, ClassInfo>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            ops: HashMap::new(),
            handlers: HashMap::new(),
            type_aliases: HashMap::new(),
            classes: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, scheme: Scheme) {
        self.values.insert(name.into(), scheme);
    }

    pub fn get(&self, name: &str) -> Option<&Scheme> {
        self.values.get(name)
    }

    fn apply_subst(&mut self, subst: &Subst) {
        self.values = self
            .values
            .iter()
            .map(|(name, scheme)| (name.clone(), scheme.apply_subst(subst)))
            .collect();
        self.classes = self
            .classes
            .iter()
            .map(|(name, info)| (name.clone(), info.apply_subst(subst)))
            .collect();
    }

    pub fn insert_op(&mut self, name: impl Into<String>, entry: OpSigEntry) -> Result<(), TypeError> {
        let name = name.into();
        if self.ops.contains_key(&name) {
            return Err(TypeError::DuplicateOp(name));
        }
        self.ops.insert(name, entry);
        Ok(())
    }

    pub fn get_op(&self, name: &str) -> Option<&OpSigEntry> {
        self.ops.get(name)
    }

    pub fn insert_handler(
        &mut self,
        name: impl Into<String>,
        info: HandlerInfo,
    ) -> Result<(), TypeError> {
        let name = name.into();
        self.handlers.insert(name, info);
        Ok(())
    }

    pub fn get_handler(&self, name: &str) -> Option<&HandlerInfo> {
        self.handlers.get(name)
    }

    pub fn insert_type_alias(
        &mut self,
        name: impl Into<String>,
        alias: TypeAlias,
    ) -> Result<(), TypeError> {
        let name = name.into();
        if self.type_aliases.contains_key(&name) {
            return Err(TypeError::DuplicateType(name));
        }
        if self.classes.contains_key(&name) {
            return Err(TypeError::DuplicateType(name));
        }
        self.type_aliases.insert(name, alias);
        Ok(())
    }

    pub fn get_type_alias(&self, name: &str) -> Option<&TypeAlias> {
        self.type_aliases.get(name)
    }

    pub fn insert_class(
        &mut self,
        name: impl Into<String>,
        info: ClassInfo,
    ) -> Result<(), TypeError> {
        let name = name.into();
        if self.classes.contains_key(&name) || self.type_aliases.contains_key(&name) {
            return Err(TypeError::DuplicateClass(name));
        }
        self.classes.insert(name, info);
        Ok(())
    }

    pub fn update_class(&mut self, name: &str, info: ClassInfo) {
        self.classes.insert(name.to_string(), info);
    }

    pub fn get_class(&self, name: &str) -> Option<&ClassInfo> {
        self.classes.get(name)
    }
}

#[derive(Debug, Default)]
struct InferState {
    subst: Subst,
    next_type: usize,
    next_row: usize,
}

impl InferState {
    fn fresh_type_var(&mut self) -> Type {
        let name = format!("t{}", self.next_type);
        self.next_type += 1;
        Type::Var(name)
    }

    fn fresh_row_var(&mut self) -> EffectRow {
        let name = format!("r{}", self.next_row);
        self.next_row += 1;
        EffectRow::Var(name)
    }
}

pub fn unify_types(subst: &mut Subst, left: &Type, right: &Type) -> Result<(), TypeError> {
    let left = subst.apply_type(left);
    let right = subst.apply_type(right);
    match (left, right) {
        (Type::Var(name), ty) | (ty, Type::Var(name)) => subst.bind_type_var(&name, &ty),
        (left, right) => {
            let (left, right) = normalize_effectful(left, right);
            match (left, right) {
                (Type::Int, Type::Int)
                | (Type::Bool, Type::Bool)
                | (Type::Unit, Type::Unit)
                | (Type::Void, Type::Void)
                | (Type::Foundation, Type::Foundation)
                | (Type::Domain, Type::Domain)
                | (Type::Ordinal, Type::Ordinal) => Ok(()),
                (Type::Class(left_name), Type::Class(right_name)) => {
                    if left_name == right_name {
                        Ok(())
                    } else {
                        Err(TypeError::Mismatch(
                            Type::Class(left_name),
                            Type::Class(right_name),
                        ))
                    }
                }
                (Type::Element(left_world), Type::Element(right_world)) => {
                    match (left_world, right_world) {
                        (Some(l), Some(r)) if l != r => Err(TypeError::Mismatch(
                            Type::Element(Some(l)),
                            Type::Element(Some(r)),
                        )),
                        _ => Ok(()),
                    }
                }
                (Type::Noetic(left), Type::Noetic(right))
                | (Type::Fractal(left), Type::Fractal(right))
                | (Type::RPM(left), Type::RPM(right))
                | (Type::QState(left), Type::QState(right)) => unify_types(subst, &left, &right),
                (Type::Fun(l1, r1), Type::Fun(l2, r2))
                | (Type::Product(l1, r1), Type::Product(l2, r2))
                | (Type::Sum(l1, r1), Type::Sum(l2, r2)) => {
                    unify_types(subst, &l1, &l2)?;
                    unify_types(subst, &r1, &r2)
                }
                (Type::Effectful(ty1, row1), Type::Effectful(ty2, row2)) => {
                    unify_types(subst, &ty1, &ty2)?;
                    unify_rows(subst, &row1, &row2)
                }
                (
                    Type::Handler {
                        effect: eff1,
                        input: in1,
                        output: out1,
                    },
                    Type::Handler {
                        effect: eff2,
                        input: in2,
                        output: out2,
                    },
                ) => {
                    unify_rows(subst, &eff1, &eff2)?;
                    unify_types(subst, &in1, &in2)?;
                    unify_types(subst, &out1, &out2)
                }
                (left, right) => Err(TypeError::Mismatch(left, right)),
            }
        }
    }
}

fn normalize_effectful(left: Type, right: Type) -> (Type, Type) {
    match (&left, &right) {
        (Type::Effectful(_, _), Type::Effectful(_, _)) => (left, right),
        (Type::Effectful(_, _), _) => {
            (left, Type::Effectful(Box::new(right), EffectRow::Empty))
        }
        (_, Type::Effectful(_, _)) => {
            (Type::Effectful(Box::new(left), EffectRow::Empty), right)
        }
        _ => (left, right),
    }
}

pub fn unify_rows(subst: &mut Subst, left: &EffectRow, right: &EffectRow) -> Result<(), TypeError> {
    let left = subst.apply_row(left);
    let right = subst.apply_row(right);
    if left == right {
        return Ok(());
    }
    match (&left, &right) {
        (EffectRow::Var(name), row) | (row, EffectRow::Var(name)) => {
            return subst.bind_row_var(name, row);
        }
        _ => {}
    }

    let (left_labels, left_tail) = flatten_row(&left);
    let (right_labels, right_tail) = flatten_row(&right);

    let mut right_set: HashSet<String> = right_labels.iter().cloned().collect();
    let mut left_only = Vec::new();
    for label in left_labels {
        if !right_set.remove(&label) {
            left_only.push(label);
        }
    }
    let right_only: Vec<String> = right_set.into_iter().collect();

    match (left_tail, right_tail) {
        (None, None) => {
            if left_only.is_empty() && right_only.is_empty() {
                Ok(())
            } else {
                Err(TypeError::RowMismatch(left, right))
            }
        }
        (Some(tail), None) => {
            if !left_only.is_empty() {
                return Err(TypeError::RowMismatch(left, right));
            }
            let row = build_row(right_only, None);
            subst.bind_row_var(&tail, &row)
        }
        (None, Some(tail)) => {
            if !right_only.is_empty() {
                return Err(TypeError::RowMismatch(left, right));
            }
            let row = build_row(left_only, None);
            subst.bind_row_var(&tail, &row)
        }
        (Some(left_tail), Some(right_tail)) => {
            if left_tail == right_tail {
                if left_only.is_empty() && right_only.is_empty() {
                    return Ok(());
                }
                return Err(TypeError::RowMismatch(left, right));
            }
            if !left_only.is_empty() && !right_only.is_empty() {
                return Err(TypeError::RowMismatch(left, right));
            }
            if left_only.is_empty() {
                let row = build_row(right_only, Some(right_tail.clone()));
                subst.bind_row_var(&left_tail, &row)
            } else {
                let row = build_row(left_only, Some(left_tail.clone()));
                subst.bind_row_var(&right_tail, &row)
            }
        }
    }
}

pub fn infer_expr(env: &TypeEnv, expr: &Expr) -> Result<InferOutput, TypeError> {
    let mut state = InferState::default();
    let mut env = env.clone();
    let out = infer_expr_inner(&mut state, &mut env, expr)?;
    Ok(InferOutput {
        ty: state.subst.apply_type(&out.ty),
        effect: state.subst.apply_row(&out.effect),
    })
}

pub fn infer_program(program: &Program) -> Result<InferProgramOutput, TypeError> {
    let mut state = InferState::default();
    let mut env = TypeEnv::new();
    let mut modules = ModuleTable::default();
    let mut decl_effect = EffectRow::Empty;

    collect_modules(&program.decls, &[], &mut modules)?;

    for decl in &program.decls {
        match decl {
            TopDecl::ModuleDecl(module) => {
                let full_path = module.path.clone();
                infer_module_exports(&mut state, &mut modules, &full_path)?;
            }
            _ => {
                let effect = infer_decl(&mut state, &mut env, decl)?;
                decl_effect = combine_effects(&mut state, &decl_effect, &effect)?;
            }
        }
    }

    let module_paths: Vec<Vec<String>> = modules.modules.keys().cloned().collect();
    for path in module_paths {
        infer_module_exports(&mut state, &mut modules, &path)?;
    }

    let entry = match &program.entry {
        Some(expr) => {
            let out = infer_expr_inner(&mut state, &mut env, expr)?;
            let effect = combine_effects(&mut state, &decl_effect, &out.effect)?;
            Some(InferOutput {
                ty: state.subst.apply_type(&out.ty),
                effect: state.subst.apply_row(&effect),
            })
        }
        None => None,
    };

    env.apply_subst(&state.subst);

    Ok(InferProgramOutput { env, entry })
}

fn collect_modules(
    decls: &[TopDecl],
    prefix: &[String],
    modules: &mut ModuleTable,
) -> Result<(), TypeError> {
    for decl in decls {
        if let TopDecl::ModuleDecl(module) = decl {
            collect_module_decl(module, prefix, modules)?;
        }
    }
    Ok(())
}

fn collect_module_decl(
    module: &ModuleDecl,
    prefix: &[String],
    modules: &mut ModuleTable,
) -> Result<(), TypeError> {
    let mut full_path = prefix.to_vec();
    full_path.extend(module.path.iter().cloned());
    let name = module_path_string(&full_path);
    if modules.modules.contains_key(&full_path) {
        return Err(TypeError::DuplicateModule(name));
    }
    modules.modules.insert(
        full_path.clone(),
        ModuleEntry {
            body: module.body.clone(),
            status: ModuleStatus::Pending,
            exports: None,
        },
    );
    collect_modules(&module.body.decls, &full_path, modules)?;
    Ok(())
}

fn infer_module_exports(
    state: &mut InferState,
    modules: &mut ModuleTable,
    path: &[String],
) -> Result<TypeEnv, TypeError> {
    let (status, body, cached_exports) = {
        let entry = modules
            .modules
            .get(path)
            .ok_or_else(|| TypeError::UnknownModule(module_path_string(path)))?;
        (entry.status, entry.body.clone(), entry.exports.clone())
    };

    match status {
        ModuleStatus::Done => {
            return Ok(cached_exports.unwrap_or_else(TypeEnv::new));
        }
        ModuleStatus::InProgress => {
            return Err(TypeError::Unimplemented("recursive modules"));
        }
        ModuleStatus::Pending => {}
    }

    if let Some(entry) = modules.modules.get_mut(path) {
        entry.status = ModuleStatus::InProgress;
    }

    let mut env = TypeEnv::new();

    for import in &body.imports {
        apply_import(state, modules, path, import, &mut env)?;
    }

    for decl in &body.decls {
        match decl {
            TopDecl::ModuleDecl(child) => {
                let child_path = extend_module_path(path, &child.path);
                infer_module_exports(state, modules, &child_path)?;
            }
            _ => {
                let _ = infer_decl(state, &mut env, decl)?;
            }
        }
    }

    env.apply_subst(&state.subst);
    let exports = match &body.exports {
        Some(export_decl) => select_exports(&env, export_decl)?,
        None => env.clone(),
    };

    if let Some(entry) = modules.modules.get_mut(path) {
        entry.exports = Some(exports.clone());
        entry.status = ModuleStatus::Done;
    }

    Ok(exports)
}

fn apply_import(
    state: &mut InferState,
    modules: &mut ModuleTable,
    current_path: &[String],
    import: &ImportDecl,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    match import {
        ImportDecl::Qualified { path, .. } => {
            let exports = resolve_module_exports(state, modules, current_path, path)?;
            merge_env(env, &exports)?;
        }
        ImportDecl::Aliased { path, .. } => {
            let exports = resolve_module_exports(state, modules, current_path, path)?;
            merge_env(env, &exports)?;
        }
        ImportDecl::Wildcard { path, .. } => {
            let exports = resolve_module_exports(state, modules, current_path, path)?;
            merge_env(env, &exports)?;
        }
        ImportDecl::Selective { path, items, .. } => {
            let exports = resolve_module_exports(state, modules, current_path, path)?;
            merge_selected(env, &exports, items)?;
        }
    }
    Ok(())
}

fn resolve_module_exports(
    state: &mut InferState,
    modules: &mut ModuleTable,
    current_path: &[String],
    path: &[String],
) -> Result<TypeEnv, TypeError> {
    if modules.modules.contains_key(path) {
        return infer_module_exports(state, modules, path);
    }
    let mut relative = current_path.to_vec();
    relative.extend(path.iter().cloned());
    if modules.modules.contains_key(&relative) {
        return infer_module_exports(state, modules, &relative);
    }
    Err(TypeError::UnknownModule(module_path_string(path)))
}

fn merge_env(dest: &mut TypeEnv, src: &TypeEnv) -> Result<(), TypeError> {
    for (name, scheme) in &src.values {
        dest.values.insert(name.clone(), scheme.clone());
    }
    for (name, entry) in &src.ops {
        dest.insert_op(name.clone(), entry.clone())?;
    }
    for (name, info) in &src.handlers {
        dest.insert_handler(name.clone(), info.clone())?;
    }
    for (name, alias) in &src.type_aliases {
        dest.insert_type_alias(name.clone(), alias.clone())?;
    }
    for (name, info) in &src.classes {
        dest.insert_class(name.clone(), info.clone())?;
    }
    Ok(())
}

fn merge_selected(
    dest: &mut TypeEnv,
    src: &TypeEnv,
    items: &[ImportItem],
) -> Result<(), TypeError> {
    for item in items {
        let target = item.alias.as_ref().unwrap_or(&item.name);
        if let Some(scheme) = src.values.get(&item.name) {
            dest.values.insert(target.clone(), scheme.clone());
        }
        if let Some(entry) = src.ops.get(&item.name) {
            dest.insert_op(target.clone(), entry.clone())?;
        }
        if let Some(info) = src.handlers.get(&item.name) {
            dest.insert_handler(target.clone(), info.clone())?;
        }
        if let Some(alias) = src.type_aliases.get(&item.name) {
            dest.insert_type_alias(target.clone(), alias.clone())?;
        }
        if let Some(info) = src.classes.get(&item.name) {
            dest.insert_class(target.clone(), info.clone())?;
        }
    }
    Ok(())
}

fn select_exports(env: &TypeEnv, export_decl: &ExportDecl) -> Result<TypeEnv, TypeError> {
    let mut out = TypeEnv::new();
    for item in &export_decl.items {
        match item {
            ExportItem::Type { name, .. } => {
                if let Some(alias) = env.type_aliases.get(name) {
                    out.insert_type_alias(name.clone(), alias.clone())?;
                }
                if let Some(info) = env.classes.get(name) {
                    out.insert_class(name.clone(), info.clone())?;
                }
            }
            ExportItem::Value(name) => {
                if let Some(scheme) = env.values.get(name) {
                    out.values.insert(name.clone(), scheme.clone());
                }
                if let Some(entry) = env.ops.get(name) {
                    out.insert_op(name.clone(), entry.clone())?;
                }
                if let Some(info) = env.handlers.get(name) {
                    out.insert_handler(name.clone(), info.clone())?;
                }
            }
        }
    }
    Ok(out)
}

fn extend_module_path(base: &[String], path: &[String]) -> Vec<String> {
    let mut out = base.to_vec();
    out.extend(path.iter().cloned());
    out
}

fn module_path_string(path: &[String]) -> String {
    if path.is_empty() {
        "<root>".to_string()
    } else {
        path.join(".")
    }
}

fn infer_decl(
    state: &mut InferState,
    env: &mut TypeEnv,
    decl: &TopDecl,
) -> Result<EffectRow, TypeError> {
    match decl {
        TopDecl::LetDecl { name, scheme, value, .. } => {
            let value_out = infer_expr_inner(state, env, value)?;
            if let Some(annotation) = scheme {
                check_annotation(state, env, annotation, &value_out)?;
            }
            env.apply_subst(&state.subst);
            let value_ty = state.subst.apply_type(&value_out.ty);
            let scheme = generalize(env, &value_ty);
            env.insert(name.clone(), scheme);
            Ok(value_out.effect)
        }
        TopDecl::EffectDecl { name, params, ops, .. } => {
            let bound: HashSet<String> = params.iter().cloned().collect();
            for op in ops {
                let input = resolve_type(env, &op.input, &bound)?;
                let output = resolve_type(env, &op.output, &bound)?;
                let entry = OpSigEntry {
                    effect: name.clone(),
                    type_vars: params.clone(),
                    input,
                    output,
                };
                env.insert_op(op.name.clone(), entry)?;
            }
            Ok(EffectRow::Empty)
        }
        TopDecl::HandlerDecl {
            name,
            handler_type,
            def,
            ..
        } => {
            let resolved_handler_type = match handler_type {
                Some(ty) => Some(resolve_type(env, ty, &HashSet::new())?),
                None => None,
            };
            let info =
                infer_handler_info(state, env, name, resolved_handler_type.as_ref(), def)?;
            env.insert_handler(name.clone(), info)?;
            Ok(EffectRow::Empty)
        }
        TopDecl::ExternDecl(decl) => {
            let mut param_types = Vec::new();
            for param in &decl.params {
                param_types.push(resolve_type(env, &param.ty, &HashSet::new())?);
            }
            let return_ty = match &decl.effects {
                Some(row) => Type::Effectful(
                    Box::new(resolve_type(env, &decl.return_type, &HashSet::new())?),
                    row.clone(),
                ),
                None => resolve_type(env, &decl.return_type, &HashSet::new())?,
            };
            let ty = build_fun_type(&param_types, return_ty);
            let scheme = generalize(env, &ty);
            env.insert(decl.name.clone(), scheme);
            Ok(EffectRow::Empty)
        }
        TopDecl::TypeDecl { name, params, body, .. } => {
            if !params.is_empty() {
                return Err(TypeError::Unimplemented("parametric type aliases"));
            }
            let alias = TypeAlias {
                params: params.clone(),
                body: body.clone(),
            };
            env.insert_type_alias(name.clone(), alias)?;
            Ok(EffectRow::Empty)
        }
        TopDecl::ClassDecl(decl) => {
            infer_class_decl(state, env, decl)?;
            Ok(EffectRow::Empty)
        }
        TopDecl::ModuleDecl(_) => Err(TypeError::Unimplemented("modules")),
    }
}

fn infer_class_decl(
    state: &mut InferState,
    env: &mut TypeEnv,
    decl: &ClassDecl,
) -> Result<(), TypeError> {
    let class_name = decl.name.clone();
    env.insert_class(class_name.clone(), ClassInfo::default())?;
    let class_ty = Type::Class(class_name.clone());

    let mut seen = HashSet::new();
    let mut info = ClassInfo::default();

    for field in &decl.fields {
        if !seen.insert(field.name.clone()) {
            return Err(TypeError::DuplicateMember {
                class: class_name.clone(),
                member: field.name.clone(),
            });
        }
        let ty = resolve_type(env, &field.ty, &HashSet::new())?;
        info.fields.insert(field.name.clone(), ty);
    }

    for prop in &decl.properties {
        if !seen.insert(prop.name.clone()) {
            return Err(TypeError::DuplicateMember {
                class: class_name.clone(),
                member: prop.name.clone(),
            });
        }
        let ty = match &prop.ty {
            Some(ty) => resolve_type(env, ty, &HashSet::new())?,
            None => state.fresh_type_var(),
        };
        let effect = state.fresh_row_var();
        info.properties
            .insert(prop.name.clone(), PropertySig { ty, effect });
    }

    for method in &decl.methods {
        if !seen.insert(method.name.clone()) {
            return Err(TypeError::DuplicateMember {
                class: class_name.clone(),
                member: method.name.clone(),
            });
        }
        let mut params = Vec::new();
        for param in &method.params {
            let ty = match &param.ty {
                Some(ty) => resolve_type(env, ty, &HashSet::new())?,
                None => state.fresh_type_var(),
            };
            params.push(ty);
        }
        let output = match &method.return_type {
            Some(ty) => resolve_type(env, ty, &HashSet::new())?,
            None => state.fresh_type_var(),
        };
        let effect = state.fresh_row_var();
        info.methods.insert(
            method.name.clone(),
            MethodSig {
                params,
                output,
                effect,
            },
        );
    }

    env.update_class(&class_name, info);

    for prop in &decl.properties {
        let info = env
            .get_class(&class_name)
            .ok_or_else(|| TypeError::UnknownClass(class_name.clone()))?;
        let sig = info
            .properties
            .get(&prop.name)
            .ok_or_else(|| TypeError::UnknownMember {
                class: class_name.clone(),
                member: prop.name.clone(),
            })?
            .clone();
        let mut local = env.clone();
        insert_self_bindings(&mut local, &class_ty);
        let out = infer_expr_inner(state, &mut local, &prop.body)?;
        unify_types(&mut state.subst, &out.ty, &sig.ty)?;
        unify_rows(&mut state.subst, &out.effect, &sig.effect)?;
        env.apply_subst(&state.subst);
    }

    for method in &decl.methods {
        let info = env
            .get_class(&class_name)
            .ok_or_else(|| TypeError::UnknownClass(class_name.clone()))?;
        let sig = info
            .methods
            .get(&method.name)
            .ok_or_else(|| TypeError::UnknownMember {
                class: class_name.clone(),
                member: method.name.clone(),
            })?
            .clone();
        let mut local = env.clone();
        insert_self_bindings(&mut local, &class_ty);
        for (idx, param) in method.params.iter().enumerate() {
            let param_ty = sig.params.get(idx).ok_or_else(|| {
                TypeError::InvalidMethodCall {
                    class: class_name.clone(),
                    method: method.name.clone(),
                }
            })?;
            local.insert(param.name.clone(), Scheme::mono(param_ty.clone()));
        }
        let out = infer_expr_inner(state, &mut local, &method.body)?;
        unify_types(&mut state.subst, &out.ty, &sig.output)?;
        unify_rows(&mut state.subst, &out.effect, &sig.effect)?;
        env.apply_subst(&state.subst);
    }

    Ok(())
}

fn insert_self_bindings(env: &mut TypeEnv, class_ty: &Type) {
    env.insert("self".to_string(), Scheme::mono(class_ty.clone()));
    env.insert("identity".to_string(), Scheme::mono(class_ty.clone()));
}

fn infer_expr_inner(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    match expr {
        Expr::Var { name, .. } => {
            let scheme = env
                .get(name)
                .ok_or_else(|| TypeError::UnknownVar(name.clone()))?;
            Ok(InferOutput {
                ty: scheme.instantiate(state),
                effect: EffectRow::Empty,
            })
        }
        Expr::Lit { literal, .. } => Ok(InferOutput {
            ty: infer_literal_type(literal)?,
            effect: EffectRow::Empty,
        }),
        Expr::Element { world, .. } => Ok(InferOutput {
            ty: Type::Element(Some(*world)),
            effect: EffectRow::Empty,
        }),
        Expr::Foundation { .. } => Ok(InferOutput {
            ty: Type::Foundation,
            effect: EffectRow::Empty,
        }),
        Expr::Lam {
            param,
            param_type,
            body,
            ..
        } => {
            let param_ty = match param_type {
                Some(ty) => extract_param_type(env, ty)?,
                None => state.fresh_type_var(),
            };
            let mut local = env.clone();
            local.insert(param.clone(), Scheme::mono(param_ty.clone()));
            let body_out = infer_expr_inner(state, &mut local, body)?;
            Ok(InferOutput {
                ty: Type::Fun(
                    Box::new(param_ty),
                    Box::new(Type::Effectful(
                        Box::new(body_out.ty),
                        body_out.effect.clone(),
                    )),
                ),
                effect: EffectRow::Empty,
            })
        }
        Expr::App { func, arg, .. } => {
            let func_out = infer_expr_inner(state, env, func)?;
            let arg_out = infer_expr_inner(state, env, arg)?;
            let result_ty = state.fresh_type_var();
            let result_eff = state.fresh_row_var();
            let expected = Type::Fun(
                Box::new(arg_out.ty.clone()),
                Box::new(Type::Effectful(
                    Box::new(result_ty.clone()),
                    result_eff.clone(),
                )),
            );
            unify_types(&mut state.subst, &func_out.ty, &expected)?;
            let effect = combine_effects(state, &func_out.effect, &arg_out.effect)?;
            let effect = combine_effects(state, &effect, &result_eff)?;
            Ok(InferOutput {
                ty: state.subst.apply_type(&result_ty),
                effect: state.subst.apply_row(&effect),
            })
        }
        Expr::Member { object, member, .. } => {
            infer_member_access(state, env, object, member)
        }
        Expr::Let {
            name,
            scheme,
            value,
            body,
            ..
        } => {
            let value_out = infer_expr_inner(state, env, value)?;
            if let Some(annotation) = scheme {
                check_annotation(state, env, annotation, &value_out)?;
            }
            env.apply_subst(&state.subst);
            let value_ty = state.subst.apply_type(&value_out.ty);
            let scheme = generalize(env, &value_ty);
            let mut local = env.clone();
            local.insert(name.clone(), scheme);
            let body_out = infer_expr_inner(state, &mut local, body)?;
            let effect = combine_effects(state, &value_out.effect, &body_out.effect)?;
            Ok(InferOutput {
                ty: body_out.ty,
                effect,
            })
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let cond_out = infer_expr_inner(state, env, cond)?;
            unify_types(&mut state.subst, &cond_out.ty, &Type::Bool)?;
            let then_out = infer_expr_inner(state, env, then_branch)?;
            let else_out = infer_expr_inner(state, env, else_branch)?;
            unify_types(&mut state.subst, &then_out.ty, &else_out.ty)?;
            let effect = combine_effects(state, &cond_out.effect, &then_out.effect)?;
            let effect = combine_effects(state, &effect, &else_out.effect)?;
            Ok(InferOutput {
                ty: state.subst.apply_type(&then_out.ty),
                effect,
            })
        }
        Expr::Noetic { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            Ok(inner)
        }
        Expr::Fractal { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            Ok(inner)
        }
        Expr::RPMReturn { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(inner.ty)),
                effect: inner.effect,
            })
        }
        Expr::RPMCheck { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            unify_types(&mut state.subst, &inner.ty, &Type::Element(None))?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(Type::Bool)),
                effect: inner.effect,
            })
        }
        Expr::RPMAcquire { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            unify_types(&mut state.subst, &inner.ty, &Type::Element(None))?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(Type::Unit)),
                effect: inner.effect,
            })
        }
        Expr::RPMBind { left, right, .. } => {
            let left_out = infer_expr_inner(state, env, left)?;
            let right_out = infer_expr_inner(state, env, right)?;
            let arg_ty = state.fresh_type_var();
            let res_ty = state.fresh_type_var();
            unify_types(&mut state.subst, &left_out.ty, &Type::RPM(Box::new(arg_ty.clone())))?;
            unify_types(
                &mut state.subst,
                &right_out.ty,
                &Type::Fun(
                    Box::new(arg_ty.clone()),
                    Box::new(Type::RPM(Box::new(res_ty.clone()))),
                ),
            )?;
            let effect = combine_effects(state, &left_out.effect, &right_out.effect)?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(res_ty)),
                effect,
            })
        }
        Expr::ACBE { goal, expr, .. } => infer_acbe(state, env, goal, expr),
        Expr::Constructor { name, args, .. } => infer_constructor(state, env, name, args),
        Expr::Handle { expr, handler, .. } => infer_handle(state, env, expr, handler),
        Expr::Perform { op, arg, .. } => infer_perform(state, env, op, arg),
        Expr::OrdLit { .. }
        | Expr::OrdOmega { .. }
        | Expr::OrdEpsilon { .. }
        | Expr::OrdAleph { .. } => Ok(InferOutput {
            ty: Type::Ordinal,
            effect: EffectRow::Empty,
        }),
        Expr::OrdSucc { expr, .. } => infer_ordinal_unary(state, env, expr),
        Expr::OrdAdd { left, right, .. }
        | Expr::OrdMul { left, right, .. }
        | Expr::OrdExp { left, right, .. } => infer_ordinal_binary(state, env, left, right),
        Expr::OrdLimit {
            binder,
            bound,
            body,
            ..
        } => {
            let bound_out = infer_expr_inner(state, env, bound)?;
            unify_types(&mut state.subst, &bound_out.ty, &Type::Ordinal)?;
            let mut local = env.clone();
            local.insert(binder.clone(), Scheme::mono(Type::Ordinal));
            let body_out = infer_expr_inner(state, &mut local, body)?;
            unify_types(&mut state.subst, &body_out.ty, &Type::Ordinal)?;
            let effect = combine_effects(state, &bound_out.effect, &body_out.effect)?;
            Ok(InferOutput {
                ty: Type::Ordinal,
                effect,
            })
        }
        Expr::TransfiniteLoop {
            index,
            ordinal,
            init,
            step_param,
            step_body,
            limit_param,
            limit_body,
            ..
        } => infer_transfinite_loop(
            state,
            env,
            index,
            ordinal,
            init,
            step_param,
            step_body,
            limit_param,
            limit_body,
        ),
        Expr::Superpose { states, .. } => infer_superpose(state, env, states),
        Expr::Measure { expr, .. } => infer_measure(state, env, expr),
        Expr::Entangle { left, right, .. } => infer_entangle(state, env, left, right),
        Expr::Ket { expr, .. } => infer_ket(state, env, expr),
        Expr::Bra { expr, .. } => infer_bra(state, env, expr),
        Expr::BraKet { left, right, .. } => infer_braket(state, env, left, right),
    }
}

fn infer_member_access(
    state: &mut InferState,
    env: &mut TypeEnv,
    object: &Expr,
    member: &str,
) -> Result<InferOutput, TypeError> {
    let object_out = infer_expr_inner(state, env, object)?;
    let object_ty = state.subst.apply_type(&object_out.ty);
    let class_name = match object_ty {
        Type::Class(name) => name,
        ty => {
            return Err(TypeError::InvalidMemberAccess {
                member: member.to_string(),
                ty,
            })
        }
    };
    let info = env
        .get_class(&class_name)
        .ok_or_else(|| TypeError::UnknownClass(class_name.clone()))?;
    if let Some(ty) = info.fields.get(member) {
        return Ok(InferOutput {
            ty: state.subst.apply_type(ty),
            effect: object_out.effect,
        });
    }
    if let Some(sig) = info.properties.get(member) {
        let sig = sig.apply_subst(&state.subst);
        let effect = combine_effects(state, &object_out.effect, &sig.effect)?;
        return Ok(InferOutput {
            ty: sig.ty,
            effect: state.subst.apply_row(&effect),
        });
    }
    if let Some(sig) = info.methods.get(member) {
        let sig = sig.apply_subst(&state.subst);
        let ty = method_value_type(&sig);
        return Ok(InferOutput {
            ty: state.subst.apply_type(&ty),
            effect: object_out.effect,
        });
    }
    Err(TypeError::UnknownMember {
        class: class_name,
        member: member.to_string(),
    })
}

fn infer_constructor(
    state: &mut InferState,
    env: &mut TypeEnv,
    name: &str,
    args: &[tkscore::ast::ConstructorArg],
) -> Result<InferOutput, TypeError> {
    let info = env
        .get_class(name)
        .ok_or_else(|| TypeError::UnknownClass(name.to_string()))?;
    let mut seen = HashSet::new();
    let mut effect = EffectRow::Empty;
    for arg in args {
        if !seen.insert(arg.name.clone()) {
            return Err(TypeError::DuplicateMember {
                class: name.to_string(),
                member: arg.name.clone(),
            });
        }
        let field_ty = info.fields.get(&arg.name).ok_or_else(|| {
            TypeError::UnknownMember {
                class: name.to_string(),
                member: arg.name.clone(),
            }
        })?;
        let value_out = infer_expr_inner(state, env, &arg.value)?;
        unify_types(&mut state.subst, &value_out.ty, field_ty)?;
        effect = combine_effects(state, &effect, &value_out.effect)?;
    }
    for field in info.fields.keys() {
        if !seen.contains(field) {
            return Err(TypeError::MissingMember {
                class: name.to_string(),
                member: field.clone(),
            });
        }
    }
    Ok(InferOutput {
        ty: Type::Class(name.to_string()),
        effect: state.subst.apply_row(&effect),
    })
}

fn infer_perform(
    state: &mut InferState,
    env: &mut TypeEnv,
    op: &str,
    arg: &Expr,
) -> Result<InferOutput, TypeError> {
    let entry = env
        .get_op(op)
        .ok_or_else(|| TypeError::UnknownOp(op.to_string()))?
        .clone();
    let (input_ty, output_ty) = instantiate_op(&entry, state);
    let arg_out = infer_expr_inner(state, env, arg)?;
    unify_types(&mut state.subst, &arg_out.ty, &input_ty)?;
    let effect = combine_effects(
        state,
        &arg_out.effect,
        &effect_row_single(&entry.effect),
    )?;
    Ok(InferOutput {
        ty: output_ty,
        effect,
    })
}

fn infer_handle(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
    handler: &HandlerRef,
) -> Result<InferOutput, TypeError> {
    let expr_out = infer_expr_inner(state, env, expr)?;
    let (effect_name, input_ty, output_ty, handler_effect) = match handler {
        HandlerRef::Named { name, .. } => {
            let info = env
                .get_handler(name)
                .ok_or_else(|| TypeError::UnknownHandler(name.clone()))?;
            let (input, output, effect_row) = info.instantiate(state);
            (info.effect.clone(), input, output, effect_row)
        }
        HandlerRef::Inline { def, .. } => {
            infer_inline_handler(state, env, def, &expr_out.ty)?
        }
    };

    unify_types(&mut state.subst, &expr_out.ty, &input_ty)?;
    let remainder = state.fresh_row_var();
    let expected = EffectRow::Cons(effect_name, Box::new(remainder.clone()));
    unify_rows(&mut state.subst, &expr_out.effect, &expected)?;
    let applied_handler_effect = state.subst.apply_row(&handler_effect);
    let (_, tail) = flatten_row(&applied_handler_effect);
    if let Some(tail) = tail {
        let tail_row = EffectRow::Var(tail);
        unify_rows(&mut state.subst, &tail_row, &remainder)?;
    }
    let effect = combine_effects(state, &remainder, &handler_effect)?;
    Ok(InferOutput {
        ty: output_ty,
        effect,
    })
}

fn infer_inline_handler(
    state: &mut InferState,
    env: &mut TypeEnv,
    def: &HandlerDef,
    input_ty: &Type,
) -> Result<(String, Type, Type, EffectRow), TypeError> {
    let effect_name = resolve_handler_effect(env, def, "<inline>")?;
    let output_ty = state.fresh_type_var();
    let handler_effect =
        infer_handler_def(state, env, def, "<inline>", &effect_name, input_ty, &output_ty)?;
    Ok((
        effect_name,
        input_ty.clone(),
        output_ty,
        handler_effect,
    ))
}

fn infer_acbe(
    state: &mut InferState,
    env: &mut TypeEnv,
    goal: &Expr,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    let goal_out = infer_expr_inner(state, env, goal)?;
    let expr_out = infer_expr_inner(state, env, expr)?;
    unify_types(&mut state.subst, &goal_out.ty, &expr_out.ty)?;
    let effect = combine_effects(state, &goal_out.effect, &expr_out.effect)?;
    Ok(InferOutput {
        ty: state.subst.apply_type(&expr_out.ty),
        effect,
    })
}

fn infer_ket(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    let inner = infer_expr_inner(state, env, expr)?;
    Ok(InferOutput {
        ty: Type::QState(Box::new(inner.ty)),
        effect: inner.effect,
    })
}

fn infer_bra(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    let inner = infer_expr_inner(state, env, expr)?;
    Ok(InferOutput {
        ty: Type::QState(Box::new(inner.ty)),
        effect: inner.effect,
    })
}

fn infer_braket(
    state: &mut InferState,
    env: &mut TypeEnv,
    left: &Expr,
    right: &Expr,
) -> Result<InferOutput, TypeError> {
    let left_out = infer_expr_inner(state, env, left)?;
    let right_out = infer_expr_inner(state, env, right)?;
    unify_types(&mut state.subst, &left_out.ty, &right_out.ty)?;
    let effect = combine_effects(state, &left_out.effect, &right_out.effect)?;
    Ok(InferOutput {
        ty: Type::Domain,
        effect,
    })
}

fn infer_superpose(
    state: &mut InferState,
    env: &mut TypeEnv,
    states: &[(Expr, Expr)],
) -> Result<InferOutput, TypeError> {
    let mut effect = EffectRow::Empty;
    let state_ty = state.fresh_type_var();
    let amp_ty = state.fresh_type_var();
    for (amp, val) in states {
        let amp_out = infer_expr_inner(state, env, amp)?;
        unify_types(&mut state.subst, &amp_out.ty, &amp_ty)?;
        effect = combine_effects(state, &effect, &amp_out.effect)?;
        let val_out = infer_expr_inner(state, env, val)?;
        let expected = Type::QState(Box::new(state_ty.clone()));
        unify_types(&mut state.subst, &val_out.ty, &expected)?;
        effect = combine_effects(state, &effect, &val_out.effect)?;
    }
    Ok(InferOutput {
        ty: Type::QState(Box::new(state.subst.apply_type(&state_ty))),
        effect,
    })
}

fn infer_measure(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    let inner = infer_expr_inner(state, env, expr)?;
    let result_ty = state.fresh_type_var();
    let expected = Type::QState(Box::new(result_ty.clone()));
    unify_types(&mut state.subst, &inner.ty, &expected)?;
    Ok(InferOutput {
        ty: state.subst.apply_type(&result_ty),
        effect: inner.effect,
    })
}

fn infer_entangle(
    state: &mut InferState,
    env: &mut TypeEnv,
    left: &Expr,
    right: &Expr,
) -> Result<InferOutput, TypeError> {
    let left_out = infer_expr_inner(state, env, left)?;
    let right_out = infer_expr_inner(state, env, right)?;
    let left_ty = state.fresh_type_var();
    let right_ty = state.fresh_type_var();
    unify_types(
        &mut state.subst,
        &left_out.ty,
        &Type::QState(Box::new(left_ty.clone())),
    )?;
    unify_types(
        &mut state.subst,
        &right_out.ty,
        &Type::QState(Box::new(right_ty.clone())),
    )?;
    let effect = combine_effects(state, &left_out.effect, &right_out.effect)?;
    Ok(InferOutput {
        ty: Type::QState(Box::new(Type::Product(
            Box::new(left_ty),
            Box::new(right_ty),
        ))),
        effect,
    })
}

fn infer_transfinite_loop(
    state: &mut InferState,
    env: &mut TypeEnv,
    index: &str,
    ordinal: &Expr,
    init: &Expr,
    step_param: &str,
    step_body: &Expr,
    limit_param: &str,
    limit_body: &Expr,
) -> Result<InferOutput, TypeError> {
    let ord_out = infer_expr_inner(state, env, ordinal)?;
    unify_types(&mut state.subst, &ord_out.ty, &Type::Ordinal)?;
    let init_out = infer_expr_inner(state, env, init)?;
    let result_ty = init_out.ty.clone();

    let mut step_env = env.clone();
    step_env.insert(step_param.to_string(), Scheme::mono(result_ty.clone()));
    step_env.insert(index.to_string(), Scheme::mono(Type::Ordinal));
    let step_out = infer_expr_inner(state, &mut step_env, step_body)?;
    unify_types(&mut state.subst, &step_out.ty, &result_ty)?;

    let mut limit_env = env.clone();
    limit_env.insert(limit_param.to_string(), Scheme::mono(result_ty.clone()));
    limit_env.insert(index.to_string(), Scheme::mono(Type::Ordinal));
    let limit_out = infer_expr_inner(state, &mut limit_env, limit_body)?;
    unify_types(&mut state.subst, &limit_out.ty, &result_ty)?;

    let effect = combine_effects(state, &ord_out.effect, &init_out.effect)?;
    let effect = combine_effects(state, &effect, &step_out.effect)?;
    let effect = combine_effects(state, &effect, &limit_out.effect)?;

    Ok(InferOutput {
        ty: state.subst.apply_type(&result_ty),
        effect,
    })
}

fn infer_handler_info(
    state: &mut InferState,
    env: &mut TypeEnv,
    name: &str,
    handler_type: Option<&Type>,
    def: &HandlerDef,
) -> Result<HandlerInfo, TypeError> {
    let (effect_name, input_ty, output_ty) = match handler_type {
        Some(ty) => match ty {
            Type::Handler {
                effect,
                input,
                output,
            } => {
                let effect_name = extract_single_effect(effect)?;
                if !def.op_clauses.is_empty() {
                    let inferred = resolve_handler_effect(env, def, name)?;
                    if inferred != effect_name {
                        return Err(TypeError::HandlerEffectMismatch {
                            handler: name.to_string(),
                            expected: effect_name,
                            found: inferred,
                        });
                    }
                }
                (effect_name, input.as_ref().clone(), output.as_ref().clone())
            }
            _ => {
                return Err(TypeError::InvalidAnnotation(format!(
                    "handler {name} annotation must be Handler[...]",
                )))
            }
        },
        None => {
            let effect_name = resolve_handler_effect(env, def, name)?;
            let input_ty = state.fresh_type_var();
            let output_ty = state.fresh_type_var();
            (effect_name, input_ty, output_ty)
        }
    };

    let handler_effect =
        infer_handler_def(state, env, def, name, &effect_name, &input_ty, &output_ty)?;
    env.apply_subst(&state.subst);

    let input_ty = state.subst.apply_type(&input_ty);
    let output_ty = state.subst.apply_type(&output_ty);
    let handler_effect = state.subst.apply_row(&handler_effect);
    let (type_vars, row_vars) = generalize_handler(env, &input_ty, &output_ty, &handler_effect);
    Ok(HandlerInfo {
        effect: effect_name,
        type_vars,
        row_vars,
        input: input_ty,
        output: output_ty,
        effect_row: handler_effect,
    })
}

fn resolve_handler_effect(
    env: &TypeEnv,
    def: &HandlerDef,
    handler_name: &str,
) -> Result<String, TypeError> {
    let mut effect = None;
    for clause in &def.op_clauses {
        let entry = env
            .get_op(&clause.op)
            .ok_or_else(|| TypeError::UnknownOp(clause.op.clone()))?;
        match &effect {
            None => {
                effect = Some(entry.effect.clone());
            }
            Some(existing) if existing != &entry.effect => {
                return Err(TypeError::HandlerEffectMismatch {
                    handler: handler_name.to_string(),
                    expected: existing.clone(),
                    found: entry.effect.clone(),
                });
            }
            _ => {}
        }
    }
    effect.ok_or_else(|| {
        TypeError::InvalidAnnotation(format!(
            "handler {handler_name} has no operations to infer effect"
        ))
    })
}

fn infer_handler_def(
    state: &mut InferState,
    env: &TypeEnv,
    def: &HandlerDef,
    handler_name: &str,
    effect_name: &str,
    input_ty: &Type,
    output_ty: &Type,
) -> Result<EffectRow, TypeError> {
    let resume_row = state.fresh_row_var();
    let mut effect = EffectRow::Empty;

    let (ret_name, ret_expr) = &def.return_clause;
    let mut ret_env = env.clone();
    ret_env.insert(ret_name.clone(), Scheme::mono(input_ty.clone()));
    let ret_out = infer_expr_inner(state, &mut ret_env, ret_expr)?;
    unify_types(&mut state.subst, &ret_out.ty, output_ty)?;
    effect = combine_effects(state, &effect, &ret_out.effect)?;

    for clause in &def.op_clauses {
        let entry = env
            .get_op(&clause.op)
            .ok_or_else(|| TypeError::UnknownOp(clause.op.clone()))?;
        if entry.effect != effect_name {
            return Err(TypeError::HandlerEffectMismatch {
                handler: handler_name.to_string(),
                expected: effect_name.to_string(),
                found: entry.effect.clone(),
            });
        }
        let (input_ty, output_ty_op) = instantiate_op(entry, state);
        let mut clause_env = env.clone();
        clause_env.insert(clause.arg.clone(), Scheme::mono(input_ty));
        let k_ty = Type::Fun(
            Box::new(output_ty_op),
            Box::new(Type::Effectful(
                Box::new(output_ty.clone()),
                resume_row.clone(),
            )),
        );
        clause_env.insert(clause.k.clone(), Scheme::mono(k_ty.clone()));
        clause_env.insert("resume".to_string(), Scheme::mono(k_ty));
        let clause_out = infer_expr_inner(state, &mut clause_env, &clause.body)?;
        unify_types(&mut state.subst, &clause_out.ty, output_ty)?;
        effect = combine_effects(state, &effect, &clause_out.effect)?;
    }

    Ok(effect)
}

fn generalize_handler(
    env: &TypeEnv,
    input: &Type,
    output: &Type,
    effect: &EffectRow,
) -> (Vec<String>, Vec<String>) {
    let mut type_vars = free_type_vars_type(input);
    type_vars.extend(free_type_vars_type(output));
    let env_type_vars = free_type_vars_env(env);
    type_vars.retain(|var| !env_type_vars.contains(var));

    let mut row_vars = free_row_vars_type(input);
    row_vars.extend(free_row_vars_type(output));
    row_vars.extend(free_row_vars_row(effect));
    let env_row_vars = free_row_vars_env(env);
    row_vars.retain(|var| !env_row_vars.contains(var));

    (
        type_vars.into_iter().collect(),
        row_vars.into_iter().collect(),
    )
}

fn extract_single_effect(effect: &EffectRow) -> Result<String, TypeError> {
    match effect {
        EffectRow::Cons(name, tail) => match tail.as_ref() {
            EffectRow::Empty => Ok(name.clone()),
            _ => Err(TypeError::InvalidAnnotation(
                "handler effect row must contain a single effect".to_string(),
            )),
        },
        _ => Err(TypeError::InvalidAnnotation(
            "handler effect row must contain a single effect".to_string(),
        )),
    }
}

fn infer_literal_type(literal: &Literal) -> Result<Type, TypeError> {
    match literal {
        Literal::Int(_) => Ok(Type::Int),
        Literal::Bool(_) => Ok(Type::Bool),
        Literal::Float(_) => Ok(Type::Domain),
        Literal::Complex { .. } => Ok(Type::Domain),
        Literal::Unit => Ok(Type::Unit),
    }
}

fn extract_param_type(env: &TypeEnv, ty: &Type) -> Result<Type, TypeError> {
    let resolved = resolve_type(env, ty, &HashSet::new())?;
    match resolved {
        Type::Effectful(_, _) => Err(TypeError::InvalidAnnotation(
            "parameter types cannot be effectful".to_string(),
        )),
        _ => Ok(resolved),
    }
}

fn infer_ordinal_unary(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    let inner = infer_expr_inner(state, env, expr)?;
    unify_types(&mut state.subst, &inner.ty, &Type::Ordinal)?;
    Ok(InferOutput {
        ty: Type::Ordinal,
        effect: inner.effect,
    })
}

fn infer_ordinal_binary(
    state: &mut InferState,
    env: &mut TypeEnv,
    left: &Expr,
    right: &Expr,
) -> Result<InferOutput, TypeError> {
    let left_out = infer_expr_inner(state, env, left)?;
    let right_out = infer_expr_inner(state, env, right)?;
    unify_types(&mut state.subst, &left_out.ty, &Type::Ordinal)?;
    unify_types(&mut state.subst, &right_out.ty, &Type::Ordinal)?;
    let effect = combine_effects(state, &left_out.effect, &right_out.effect)?;
    Ok(InferOutput {
        ty: Type::Ordinal,
        effect,
    })
}

fn combine_effects(
    state: &mut InferState,
    left: &EffectRow,
    right: &EffectRow,
) -> Result<EffectRow, TypeError> {
    let left = state.subst.apply_row(left);
    let right = state.subst.apply_row(right);
    if left == EffectRow::Empty {
        return Ok(right);
    }
    if right == EffectRow::Empty {
        return Ok(left);
    }
    let (mut labels_left, tail_left) = flatten_row(&left);
    let (labels_right, tail_right) = flatten_row(&right);
    labels_left.extend(labels_right);
    let tail = match (tail_left, tail_right) {
        (None, None) => None,
        (Some(tail), None) | (None, Some(tail)) => Some(tail),
        (Some(left_tail), Some(right_tail)) => {
            if left_tail != right_tail {
                let left_var = EffectRow::Var(left_tail.clone());
                let right_var = EffectRow::Var(right_tail.clone());
                unify_rows(&mut state.subst, &left_var, &right_var)?;
            }
            Some(left_tail)
        }
    };
    Ok(build_row(labels_left, tail))
}

fn effect_row_single(name: &str) -> EffectRow {
    EffectRow::Cons(name.to_string(), Box::new(EffectRow::Empty))
}

fn instantiate_op(entry: &OpSigEntry, state: &mut InferState) -> (Type, Type) {
    let mut subst = Subst::default();
    for var in &entry.type_vars {
        subst.type_subst.insert(var.clone(), state.fresh_type_var());
    }
    (subst.apply_type(&entry.input), subst.apply_type(&entry.output))
}

fn check_annotation(
    state: &mut InferState,
    env: &TypeEnv,
    annotation: &TypeScheme,
    out: &InferOutput,
) -> Result<(), TypeError> {
    let scheme = scheme_from_annotation(env, annotation)?;
    let inst = scheme.instantiate(state);
    match inst {
        Type::Effectful(inner, row) => {
            unify_types(&mut state.subst, &inner, &out.ty)?;
            unify_rows(&mut state.subst, &row, &out.effect)
        }
        _ => {
            unify_types(&mut state.subst, &inst, &out.ty)?;
            unify_rows(&mut state.subst, &out.effect, &EffectRow::Empty)
        }
    }
}

fn build_fun_type(param_types: &[Type], return_type: Type) -> Type {
    param_types.iter().rev().fold(return_type, |acc, ty| {
        Type::Fun(Box::new(ty.clone()), Box::new(acc))
    })
}

fn method_value_type(sig: &MethodSig) -> Type {
    let ret = Type::Effectful(Box::new(sig.output.clone()), sig.effect.clone());
    build_fun_type(&sig.params, ret)
}

fn scheme_from_annotation(env: &TypeEnv, annotation: &TypeScheme) -> Result<Scheme, TypeError> {
    let bound: HashSet<String> = annotation.vars.iter().cloned().collect();
    let resolved = resolve_type(env, &annotation.ty, &bound)?;
    let row_vars = free_row_vars_type(&resolved);
    Ok(Scheme {
        type_vars: annotation.vars.clone(),
        row_vars: row_vars.into_iter().collect(),
        ty: resolved,
    })
}

fn resolve_type(
    env: &TypeEnv,
    ty: &Type,
    bound: &HashSet<String>,
) -> Result<Type, TypeError> {
    let mut seen = HashSet::new();
    resolve_type_inner(env, ty, bound, &mut seen)
}

fn resolve_type_inner(
    env: &TypeEnv,
    ty: &Type,
    bound: &HashSet<String>,
    seen: &mut HashSet<String>,
) -> Result<Type, TypeError> {
    match ty {
        Type::Var(name) => {
            if bound.contains(name) {
                return Ok(Type::Var(name.clone()));
            }
            if env.get_class(name).is_some() {
                return Ok(Type::Class(name.clone()));
            }
            if let Some(alias) = env.get_type_alias(name) {
                if !alias.params.is_empty() {
                    return Err(TypeError::Unimplemented("parametric type aliases"));
                }
                if !seen.insert(name.clone()) {
                    return Err(TypeError::RecursiveTypeAlias(name.clone()));
                }
                let resolved = resolve_type_inner(env, &alias.body, bound, seen)?;
                seen.remove(name);
                return Ok(resolved);
            }
            Ok(Type::Var(name.clone()))
        }
        Type::Noetic(inner) => Ok(Type::Noetic(Box::new(resolve_type_inner(
            env, inner, bound, seen,
        )?))),
        Type::Fractal(inner) => Ok(Type::Fractal(Box::new(resolve_type_inner(
            env, inner, bound, seen,
        )?))),
        Type::RPM(inner) => Ok(Type::RPM(Box::new(resolve_type_inner(
            env, inner, bound, seen,
        )?))),
        Type::QState(inner) => Ok(Type::QState(Box::new(resolve_type_inner(
            env, inner, bound, seen,
        )?))),
        Type::Fun(left, right) => Ok(Type::Fun(
            Box::new(resolve_type_inner(env, left, bound, seen)?),
            Box::new(resolve_type_inner(env, right, bound, seen)?),
        )),
        Type::Product(left, right) => Ok(Type::Product(
            Box::new(resolve_type_inner(env, left, bound, seen)?),
            Box::new(resolve_type_inner(env, right, bound, seen)?),
        )),
        Type::Sum(left, right) => Ok(Type::Sum(
            Box::new(resolve_type_inner(env, left, bound, seen)?),
            Box::new(resolve_type_inner(env, right, bound, seen)?),
        )),
        Type::Effectful(inner, row) => Ok(Type::Effectful(
            Box::new(resolve_type_inner(env, inner, bound, seen)?),
            row.clone(),
        )),
        Type::Handler {
            effect,
            input,
            output,
        } => Ok(Type::Handler {
            effect: effect.clone(),
            input: Box::new(resolve_type_inner(env, input, bound, seen)?),
            output: Box::new(resolve_type_inner(env, output, bound, seen)?),
        }),
        Type::Class(name) => Ok(Type::Class(name.clone())),
        _ => Ok(ty.clone()),
    }
}

fn generalize(env: &TypeEnv, ty: &Type) -> Scheme {
    let mut type_vars = free_type_vars_type(ty);
    let env_type_vars = free_type_vars_env(env);
    type_vars.retain(|var| !env_type_vars.contains(var));

    let mut row_vars = free_row_vars_type(ty);
    let env_row_vars = free_row_vars_env(env);
    row_vars.retain(|var| !env_row_vars.contains(var));

    Scheme {
        type_vars: type_vars.into_iter().collect(),
        row_vars: row_vars.into_iter().collect(),
        ty: ty.clone(),
    }
}

fn free_type_vars_env(env: &TypeEnv) -> HashSet<String> {
    let mut vars: HashSet<String> = env
        .values
        .values()
        .flat_map(|scheme| free_type_vars_scheme(scheme))
        .collect();
    for info in env.classes.values() {
        vars.extend(free_type_vars_class(info));
    }
    vars
}

fn free_row_vars_env(env: &TypeEnv) -> HashSet<String> {
    let mut vars: HashSet<String> = env
        .values
        .values()
        .flat_map(|scheme| free_row_vars_scheme(scheme))
        .collect();
    for info in env.classes.values() {
        vars.extend(free_row_vars_class(info));
    }
    vars
}

fn free_type_vars_scheme(scheme: &Scheme) -> HashSet<String> {
    let mut vars = free_type_vars_type(&scheme.ty);
    for var in &scheme.type_vars {
        vars.remove(var);
    }
    vars
}

fn free_row_vars_scheme(scheme: &Scheme) -> HashSet<String> {
    let mut vars = free_row_vars_type(&scheme.ty);
    for var in &scheme.row_vars {
        vars.remove(var);
    }
    vars
}

fn free_type_vars_type(ty: &Type) -> HashSet<String> {
    match ty {
        Type::Var(name) => HashSet::from([name.clone()]),
        Type::Noetic(inner)
        | Type::Fractal(inner)
        | Type::RPM(inner)
        | Type::QState(inner) => free_type_vars_type(inner),
        Type::Fun(left, right)
        | Type::Product(left, right)
        | Type::Sum(left, right) => {
            let mut vars = free_type_vars_type(left);
            vars.extend(free_type_vars_type(right));
            vars
        }
        Type::Effectful(inner, _) => free_type_vars_type(inner),
        Type::Handler { input, output, .. } => {
            let mut vars = free_type_vars_type(input);
            vars.extend(free_type_vars_type(output));
            vars
        }
        _ => HashSet::new(),
    }
}

fn free_row_vars_type(ty: &Type) -> HashSet<String> {
    match ty {
        Type::Effectful(inner, row) => {
            let mut vars = free_row_vars_row(row);
            vars.extend(free_row_vars_type(inner));
            vars
        }
        Type::Handler {
            effect,
            input,
            output,
        } => {
            let mut vars = free_row_vars_row(effect);
            vars.extend(free_row_vars_type(input));
            vars.extend(free_row_vars_type(output));
            vars
        }
        Type::Noetic(inner)
        | Type::Fractal(inner)
        | Type::RPM(inner)
        | Type::QState(inner) => free_row_vars_type(inner),
        Type::Fun(left, right)
        | Type::Product(left, right)
        | Type::Sum(left, right) => {
            let mut vars = free_row_vars_type(left);
            vars.extend(free_row_vars_type(right));
            vars
        }
        _ => HashSet::new(),
    }
}

fn free_row_vars_row(row: &EffectRow) -> HashSet<String> {
    match row {
        EffectRow::Empty => HashSet::new(),
        EffectRow::Cons(_, tail) => free_row_vars_row(tail),
        EffectRow::Var(name) => HashSet::from([name.clone()]),
    }
}

fn free_type_vars_class(info: &ClassInfo) -> HashSet<String> {
    let mut vars = HashSet::new();
    for ty in info.fields.values() {
        vars.extend(free_type_vars_type(ty));
    }
    for sig in info.properties.values() {
        vars.extend(free_type_vars_type(&sig.ty));
    }
    for sig in info.methods.values() {
        for param in &sig.params {
            vars.extend(free_type_vars_type(param));
        }
        vars.extend(free_type_vars_type(&sig.output));
    }
    vars
}

fn free_row_vars_class(info: &ClassInfo) -> HashSet<String> {
    let mut vars = HashSet::new();
    for sig in info.properties.values() {
        vars.extend(free_row_vars_row(&sig.effect));
    }
    for sig in info.methods.values() {
        vars.extend(free_row_vars_row(&sig.effect));
    }
    vars
}

fn type_contains_var(ty: &Type, name: &str, subst: &Subst) -> bool {
    match subst.apply_type(ty) {
        Type::Var(var) => var == name,
        Type::Noetic(inner)
        | Type::Fractal(inner)
        | Type::RPM(inner)
        | Type::QState(inner) => type_contains_var(&inner, name, subst),
        Type::Fun(left, right)
        | Type::Product(left, right)
        | Type::Sum(left, right) => {
            type_contains_var(&left, name, subst) || type_contains_var(&right, name, subst)
        }
        Type::Effectful(inner, row) => {
            type_contains_var(&inner, name, subst) || row_contains_var(&row, name, subst)
        }
        Type::Handler {
            effect,
            input,
            output,
        } => {
            row_contains_var(&effect, name, subst)
                || type_contains_var(&input, name, subst)
                || type_contains_var(&output, name, subst)
        }
        _ => false,
    }
}

fn row_contains_var(row: &EffectRow, name: &str, subst: &Subst) -> bool {
    match subst.apply_row(row) {
        EffectRow::Empty => false,
        EffectRow::Cons(_, tail) => row_contains_var(&tail, name, subst),
        EffectRow::Var(var) => var == name,
    }
}

fn flatten_row(row: &EffectRow) -> (Vec<String>, Option<String>) {
    let mut labels = Vec::new();
    let mut tail = None;
    let mut current = row;
    loop {
        match current {
            EffectRow::Empty => break,
            EffectRow::Var(name) => {
                tail = Some(name.clone());
                break;
            }
            EffectRow::Cons(name, next) => {
                labels.push(name.clone());
                current = next;
            }
        }
    }
    labels.sort();
    labels.dedup();
    (labels, tail)
}

fn build_row(mut labels: Vec<String>, tail: Option<String>) -> EffectRow {
    labels.sort();
    labels.dedup();
    let mut row = match tail {
        Some(name) => EffectRow::Var(name),
        None => EffectRow::Empty,
    };
    for label in labels.into_iter().rev() {
        row = EffectRow::Cons(label, Box::new(row));
    }
    row
}
